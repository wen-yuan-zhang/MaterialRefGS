#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import torch
import torch.nn.functional as F
import math
from arguments.config import FLAG,INDIRECT_TYPE
if FLAG == "2dgs":
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
else:
    from diff_surfel_rasterization2 import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.env_gaussian_model import EnvGaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.refl_utils import  get_specular_color_surfel2, get_specular_color_surfel,get_full_color_volume, get_full_color_volume_indirect, get_specular_color_surfel_with_raytracing_indirect,safe_normalize,sample_camera_rays_unnormalize,reflection,get_specular_color_surfel4
from utils.graphics_utils import linear_to_srgb, srgb_to_linear, rotation_between_z, init_predefined_omega
from gaussian_renderer.optix_utils import HardwareRendering
import numpy as np

def get_distance(scaling_modifier, means3D, viewpoint_camera, pc):
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    
    global_normal = pc.get_normal(scaling_modifier, dir_pp_normalized)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    return local_distance.unsqueeze(-1)

def compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe, return_depth_normal=True):
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    if FLAG == "2dgs":
        surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    else:
        render_depth_unbiased = allmap[7:8]
        render_depth_unbiased = torch.nan_to_num(render_depth_unbiased, 0, 0)
        surf_depth = render_depth_unbiased

    if return_depth_normal:
        # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        surf_normal = surf_normal * render_alpha.detach()
    else:
        surf_normal = None
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    
    
    return {
        'render_alpha': render_alpha,
        'render_normal': render_normal,
        'render_depth_median': render_depth_median,
        'render_depth_expected': render_depth_expected,
        'render_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal
    }



def render_initial(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None):

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    local_distance = get_distance(scaling_modifier,means3D,viewpoint_camera,pc)
    assert local_distance.shape == torch.Size((means3D.shape[0],1))

    if FLAG == "2dgs":
        features = torch.empty((local_distance.shape[0],0))
    else:
        features = local_distance

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_depth_median = regularizations['render_depth_median']
    render_depth_expected = regularizations['render_depth_expected']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # Transform linear rgb to srgb with nonlinearly distribution between 0 to 1
    if srgb: 
        rendered_image = linear_to_srgb(rendered_image)
    final_image = rendered_image + bg_color[:, None, None] * (1 - render_alpha)

    rets =  {"render": final_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
    }

    if FLAG != "2dgs":
        rets.update({
            "rend_distance": rendered_features[0:1,:,:],
        })

    return rets




def render_surfel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None, wo_render_img=False, normal_img_map=None):

 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    ## reflection strength 定义（即refl ratio）
    refl = pc.get_refl
    specular_weight = pc.get_specular
    ori_color = pc.get_ori_color
    roughness = pc.get_rough


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # indirect light
    if pipe.use_asg:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        splat2world = pc.get_covariance(scaling_modifier)
        normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
        w_o = -dir_pp_normalized
        reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
        
        rotation_normal = rotation_between_z(normals).transpose(-1, -2)
        reflection_cartesian = (rotation_normal @ reflection[..., None])[..., 0]
        
        # import pdb;pdb.set_trace()
        omega, omega_la, omega_mu = pc.asg_param
        asg = pc.get_asg
        ep, la, mu = torch.split(asg, [3, 1, 1], dim=-1)
        
        Smooth = F.relu((reflection_cartesian[:, None] * omega[None]).sum(dim=-1, keepdim=True))

        ep = torch.exp(ep-3)
        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)
        exp_input = -la * (omega_la[None] * reflection_cartesian[:, None]).sum(dim=-1, keepdim=True).pow(2) - mu * (omega_mu[None] * reflection_cartesian[:, None]).sum(dim=-1, keepdim=True).pow(2)
        indirect_asg = ep * Smooth * torch.exp(exp_input)
        indirect = indirect_asg.sum(dim=1).clamp_min(0.0)
    else:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
        w_o = -dir_pp_normalized
        reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
        # import pdb;pdb.set_trace()
        shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        sh2indirect = eval_sh(3, shs_indirect, reflection)
        indirect = torch.clamp_min(sh2indirect, 0.0)
    
    local_distance = get_distance(scaling_modifier,means3D,viewpoint_camera,pc)
    assert local_distance.shape == torch.Size((means3D.shape[0],1))

    
    if FLAG == "2dgs":
        features = torch.cat((refl, roughness, ori_color, indirect), dim=-1)
    else:
        features = torch.cat((refl, roughness, ori_color, indirect, specular_weight, local_distance), dim=-1)

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )


    base_color = rendered_image
    refl_strength = rendered_features[:1]
    roughness = rendered_features[1:2]
    albedo = rendered_features[2:5]
    indirect_light = rendered_features[5:8]
    blend_weight = rendered_features[8:9]

    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe, return_depth_normal=(not wo_render_img))
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']


    if wo_render_img:
        results =  {
                "refl_strength_map": refl_strength,
                "base_color_map": albedo,
                "roughness_map": roughness,
                "viewspace_points": means2D,
                "visibility_filter" : radii > 0,
                "radii": radii,
                ## normal, accum alpha, dist, depth map
                'rend_alpha': render_alpha,
                'rend_normal': render_normal,
                'rend_dist': render_dist,
                'surf_depth': surf_depth,
                'surf_normal': surf_normal,
                "blend_weight": blend_weight
        }
        if FLAG != "2dgs":
            results.update({
                "rend_distance": rendered_features[-1:,:,:],
            })
        return results 
                

    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)

    gt=viewpoint_camera.get_image()[0]
    if opt.indirect:
        specular, extra_dict = get_specular_color_surfel2(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), refl_strength=refl_strength.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth, indirect_light=indirect_light.permute(1,2,0))
    else:
        specular, extra_dict = get_specular_color_surfel2(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), refl_strength=refl_strength.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth)
    # import ipdb;ipdb.set_trace()
    # Integrate the final image
    final_image = (1-refl_strength) * base_color + specular 

    
    # Transform linear rgb to srgb with nonlinearly distribution between 0 to 1
    if srgb: 
        final_image = linear_to_srgb(final_image)
        albedo = linear_to_srgb(albedo)
        specular = linear_to_srgb(specular)


    final_image = final_image + bg_color[:, None, None] * (1 - render_alpha)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {"render": final_image,
            "refl_strength_map": refl_strength,
            "diffuse_map": (1-refl_strength) * base_color,
            "diffuse_map_ori": base_color,
            "specular_map": specular,
            "base_color_map": albedo,
            "roughness_map": roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'blend_weight': blend_weight
    }
    
    if opt.indirect:
        results.update(extra_dict)

    if FLAG != "2dgs":
        results.update({
            "rend_distance": rendered_features[-1:,...],
        })

    return results


def render_surfel2(indirect_renderer:HardwareRendering, env : EnvGaussianModel, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None, wo_render_img=False, normal_img_map=None):

 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    ## reflection strength 定义（即refl ratio）
    refl = pc.get_refl
    blend_weight = pc.get_specular
    ori_color = pc.get_ori_color
    roughness = pc.get_rough


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # indirect light
    if pipe.use_asg:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        splat2world = pc.get_covariance(scaling_modifier)
        normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
        w_o = -dir_pp_normalized
        reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
        
        rotation_normal = rotation_between_z(normals).transpose(-1, -2)
        reflection_cartesian = (rotation_normal @ reflection[..., None])[..., 0]
        
        # import pdb;pdb.set_trace()
        omega, omega_la, omega_mu = pc.asg_param
        asg = pc.get_asg
        ep, la, mu = torch.split(asg, [3, 1, 1], dim=-1)
        
        Smooth = F.relu((reflection_cartesian[:, None] * omega[None]).sum(dim=-1, keepdim=True))

        ep = torch.exp(ep-3)
        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)
        exp_input = -la * (omega_la[None] * reflection_cartesian[:, None]).sum(dim=-1, keepdim=True).pow(2) - mu * (omega_mu[None] * reflection_cartesian[:, None]).sum(dim=-1, keepdim=True).pow(2)
        indirect_asg = ep * Smooth * torch.exp(exp_input)
        indirect = indirect_asg.sum(dim=1).clamp_min(0.0)
    else:
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
        w_o = -dir_pp_normalized
        reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
        # import pdb;pdb.set_trace()
        shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        sh2indirect = eval_sh(3, shs_indirect, reflection)
        indirect = torch.clamp_min(sh2indirect, 0.0)
    
    local_distance = get_distance(scaling_modifier,means3D,viewpoint_camera,pc)
    assert local_distance.shape == torch.Size((means3D.shape[0],1))

    
    if FLAG == "2dgs":
        features = torch.cat((refl, roughness, ori_color, indirect), dim=-1)
    else:
        features = torch.cat((refl, roughness, ori_color, indirect, blend_weight, local_distance), dim=-1)

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )


    base_color = rendered_image
    refl_strength = rendered_features[:1]
    roughness = rendered_features[1:2]
    albedo = rendered_features[2:5]
    indirect_light_residual = rendered_features[5:8]
    blend_weight = rendered_features[8:9]

    # albedo = albedo.clamp(.98,1.)
    # refl_strength = refl_strength.clamp(.8,1.)
    # roughness = roughness.clamp(.0,.2)
    # opt.indirect = False

    # albedo = albedo.clamp(.98,1.)
    # refl_strength = refl_strength.clamp(.95,1.)
    # roughness = roughness.clamp(.0,.1)
    # opt.indirect = False


    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe, return_depth_normal=(not wo_render_img))
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']


    if wo_render_img:
        results =  {
                "refl_strength_map": refl_strength,
                "base_color_map": albedo,
                "roughness_map": roughness,
                "viewspace_points": means2D,
                "visibility_filter" : radii > 0,
                "radii": radii,
                ## normal, accum alpha, dist, depth map
                'rend_alpha': render_alpha,
                'rend_normal': render_normal,
                'rend_dist': render_dist,
                'surf_depth': surf_depth,
                'surf_normal': surf_normal,
                "blend_weight": blend_weight
        }
        if FLAG != "2dgs":
            results.update({
                "rend_distance": rendered_features[-1:,:,:],
            })
        return results 
                

    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)

    indirect_results = render_indirect(indirect_renderer,viewpoint_camera,env,pipe,bg_color,normal_map,surf_depth)
    indirect_light = indirect_results["render"]
    use_indirect_light_residual = False

    gt=viewpoint_camera.get_image()[0]
    if opt.indirect:
        specular, extra_dict = get_specular_color_surfel4(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), refl_strength=refl_strength.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth, indirect_light=indirect_light.permute(1,2,0), indirect_light_residual=indirect_light_residual.permute(1,2,0),blend_weight=blend_weight.permute(1,2,0),use_indirect_light_residual=use_indirect_light_residual
    )
    else:
        specular, extra_dict = get_specular_color_surfel(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), refl_strength=refl_strength.permute(1,2,0), roughness=roughness.permute(1,2,0), pc=pc, surf_depth=surf_depth)
    # import ipdb;ipdb.set_trace()
    # Integrate the final image
    final_image = (1-refl_strength) * base_color + specular 

    
    # Transform linear rgb to srgb with nonlinearly distribution between 0 to 1
    if srgb: 
        final_image = linear_to_srgb(final_image)
        albedo = linear_to_srgb(albedo)
        specular = linear_to_srgb(specular)


    final_image = final_image + bg_color[:, None, None] * (1 - render_alpha)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {"render": final_image,
            "refl_strength_map": refl_strength,
            "diffuse_map": (1-refl_strength) * base_color,
            "diffuse_map_ori": base_color,
            "specular_map": specular,
            "base_color_map": albedo,
            "roughness_map": roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'blend_weight': blend_weight,
            'indirect_out': indirect_results
    }
    
    if opt.indirect:
        results.update(extra_dict)

    if FLAG != "2dgs":
        results.update({
            "rend_distance": rendered_features[-1:,...],
        })

    return results


def render_indirect(indirect_renderer:HardwareRendering, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, normal_map:torch.Tensor=None,surf_depth:torch.Tensor=None):
    H,W,_ = viewpoint_camera.HWK
    rays_cam, rays_o = sample_camera_rays_unnormalize(viewpoint_camera.HWK,viewpoint_camera.R,viewpoint_camera.T)
    intersections = rays_o + (surf_depth.reshape(H,W,1)) * rays_cam

    w_o = safe_normalize(-rays_cam)
    rays_refl,_ = reflection(w_o,normal_map.reshape((H,W,3)))
    rays_refl = safe_normalize(rays_refl)
    intersections = intersections + (1e-3) * rays_refl

    results_indirect = indirect_renderer.render_gaussians(
        viewpoint_camera,
        ray_o = intersections, ray_d = rays_refl,
        pcd = pc, pipe = pipe, bg_color = bg_color, start_from_first = True
    )
    return results_indirect

def render_surfel_with_envgs(indirect_renderer:HardwareRendering, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None, wo_render_img=False, normal_img_map=None):
    results = render_surfel(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, srgb, opt, wo_render_img, normal_img_map
    )
    final_image = results["render"]
    surf_depth = results["surf_depth"]
    render_normal = results["rend_normal"].permute(1,2,0)
    render_alpha = results["rend_alpha"].permute(1,2,0)
    blend_weight = results["blend_weight"]
    normal_map = render_normal / render_alpha.clamp_min(1e-6)
    normal_map = safe_normalize(normal_map)

    H,W,_ = viewpoint_camera.HWK
    rays_cam, rays_o = sample_camera_rays_unnormalize(viewpoint_camera.HWK,viewpoint_camera.R,viewpoint_camera.T)
    intersections = rays_o + (surf_depth.reshape(H,W,1)) * rays_cam

    w_o = safe_normalize(-rays_cam)
    rays_refl,_ = reflection(w_o,normal_map.reshape((H,W,3)))
    rays_refl = safe_normalize(rays_refl)
    intersections = intersections + (1e-3) * rays_refl

    results_indirect = indirect_renderer.render_gaussians(
        viewpoint_camera,
        ray_o = intersections, ray_d = rays_refl,
        pcd = pc, pipe = pipe, bg_color = bg_color, start_from_first = True
    )
    results_indirect["specular"] = blend_weight
    indirect_light = results_indirect["render"]
    
    final_image = final_image * (1-blend_weight)  + blend_weight * indirect_light

    results["render"] = final_image
    results.update({
        "indirect_out":results_indirect
    })
    return results


def render_surfel_with_envgs_sep(indirect_renderer:HardwareRendering,env : EnvGaussianModel ,viewpoint_camera, pc : GaussianModel,pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, srgb = False, opt=None, wo_render_img=False, normal_img_map=None):
    results = render_surfel(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, srgb, opt, wo_render_img, normal_img_map
    )
    final_image = results["render"]
    surf_depth = results["surf_depth"]
    render_normal = results["rend_normal"].permute(1,2,0)
    render_alpha = results["rend_alpha"].permute(1,2,0)
    specular_weight = results["specular_weight"]
    normal_map = render_normal / render_alpha.clamp_min(1e-6)
    normal_map = safe_normalize(normal_map)

    H,W,_ = viewpoint_camera.HWK
    rays_cam, rays_o = sample_camera_rays_unnormalize(viewpoint_camera.HWK,viewpoint_camera.R,viewpoint_camera.T)
    intersections = rays_o + (surf_depth.reshape(H,W,1)) * rays_cam

    w_o = safe_normalize(-rays_cam)
    rays_refl,_ = reflection(w_o,normal_map.reshape((H,W,3)))
    rays_refl = safe_normalize(rays_refl)
    intersections = intersections + (1e-3) * rays_refl

    results_indirect = indirect_renderer.render_gaussians(
        viewpoint_camera,
        ray_o = intersections, ray_d = rays_refl,
        pcd = env, pipe = pipe, bg_color = bg_color, start_from_first = True
    )
    results_indirect["specular"] = specular_weight
    indirect_light = results_indirect["render"]

    final_image = final_image * (1-specular_weight)  + specular_weight * indirect_light

    results["render"] = final_image
    results.update({
        "indirect_out":results_indirect
    })
    return results

