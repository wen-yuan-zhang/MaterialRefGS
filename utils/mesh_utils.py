#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
import torch.utils
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from utils.image_utils import visualize_d_mask,visualize_depth
from functools import partial
import open3d as o3d
import trimesh
import cv2
from torchvision.utils import save_image,make_grid
import torch.functional as F
from plyfile import PlyData,PlyElement
import imageio.v2 as imageio
from utils.refl_utils import safe_normalize

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []
        self.all_maps = []
        self.diffuse_maps = []
        self.base_color_maps = []
        self.metallic_maps = []
        self.rougness_maps = []
        self.normal_prior = []

    @torch.no_grad()
    def get_normal_prior(self,viewpoint_cam):
        ROOT = os.getcwd().split("/")[1]
        import ipdb;ipdb.set_trace()
        (H,W,_) = viewpoint_cam.HWK 
        img_name = viewpoint_cam.image_name
        npr_img = cv2.imread(os.path.join(f"/{ROOT}/tjm/code/Metric3D/output/gardenspheres/normal",f"{img_name}.png"))
        npr_img = cv2.resize(npr_img,(W,H))
        npr_map = torch.from_numpy((npr_img / 255 * 2. - 1.))[...,:3].float().cuda()
        npr_map = safe_normalize(npr_map)
        npr_map_ = npr_map@(viewpoint_cam.world_view_transform[:3,:3].T)
        return npr_map_.detach().cpu().permute((2,0,1))

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, opt=None):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack



        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            # import ipdb;ipdb.set_trace()
            render_pkg = self.render(viewpoint_cam, self.gaussians, opt = opt)
            if viewpoint_cam.gt_alpha_mask is not None:
                gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            else:
                gt_alpha_mask = torch.ones_like(render_pkg['surf_depth']).cuda()
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth = depth * gt_alpha_mask + (1-gt_alpha_mask) * torch.ones_like(depth) * 10
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
            # self.normal_prior.append(self.get_normal_prior(viewpoint_cam))

            error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])
            rot = viewpoint_cam.R.T.float().cuda()
            
            def a(x):
                sz=x.shape
                x = (rot@x.reshape(3,-1)).reshape(*sz)[[2,1,0],...]
                return 0.5 * x + 0.5

            # visual_list  =   [
            #     viewpoint_cam.original_image.cuda(),  
            #     render_pkg["render"],  
            #     render_pkg["base_color_map"],  
            #     render_pkg["diffuse_map"],
            #     render_pkg["specular_map"],
            #     render_pkg["refl_strength_map"].repeat(3, 1, 1),  
            #     render_pkg["roughness_map"].repeat(3, 1, 1),
            #     render_pkg["rend_alpha"].repeat(3, 1, 1),  
            #     visualize_depth(render_pkg["surf_depth"]),  
            #     visualize_depth(render_pkg["rend_distance"]),
            #     a(render_pkg["rend_normal"]),  
            #     a(render_pkg["surf_normal"]),  
            #     error_map, 
            # ]
            # self.base_color_maps.append(render_pkg["base_color_map"].detach().cpu())
            # self.diffuse_maps.append(render_pkg["diffuse_map"].detach().cpu())
            # self.rougness_maps.append(render_pkg["roughness_map"].detach().cpu().repeat(3, 1, 1))
            # self.metallic_maps.append(render_pkg["refl_strength_map"].detach().cpu().repeat(3, 1, 1))

            # if "rend_distance" in render_pkg.keys():
            #     visual_list.append(visualize_depth(render_pkg["rend_distance"]))   
            # for idx in range(len(visual_list)):
            #     visual_list[idx] = visual_list[idx].detach().cpu()
            # self.all_maps.append(visual_list)

        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True,rgbmaps=None):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i] if rgbmaps is None else rgbmaps[i]
            depth = self.depthmaps[i]
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh
    
    def extract_mesh_bouned_with_material(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True,save_path=None):
        mesh_normal = self.extract_mesh_bounded(voxel_size,sdf_trunc,depth_trunc,mask_backgrond,rgbmaps=[_*.5+.5 for _ in self.normal_prior])
        mesh_rgb = self.extract_mesh_bounded(voxel_size,sdf_trunc,depth_trunc,mask_backgrond,rgbmaps=self.rgbmaps)
        mesh_diffuse = self.extract_mesh_bounded(voxel_size,sdf_trunc,depth_trunc,mask_backgrond,rgbmaps=self.diffuse_maps)
        mesh_basecolor = self.extract_mesh_bounded(voxel_size,sdf_trunc,depth_trunc,mask_backgrond,rgbmaps=self.base_color_maps)
        mesh_metallic = self.extract_mesh_bounded(voxel_size,sdf_trunc,depth_trunc,mask_backgrond,rgbmaps=self.metallic_maps)
        mesh_roughness = self.extract_mesh_bounded(voxel_size,sdf_trunc,depth_trunc,mask_backgrond,rgbmaps=self.rougness_maps)

        mesh_rgb = post_process_mesh(mesh_rgb,1)
        mesh_normal = post_process_mesh(mesh_normal,1)
        mesh_diffuse = post_process_mesh(mesh_diffuse,1)
        mesh_basecolor = post_process_mesh(mesh_basecolor,1)
        mesh_metallic = post_process_mesh(mesh_metallic,1)
        mesh_roughness = post_process_mesh(mesh_roughness,1)

        mesh_lst = [mesh_rgb,mesh_normal,mesh_diffuse,mesh_basecolor,mesh_metallic,mesh_roughness]
        for u in mesh_lst:
            for v in mesh_lst:
                assert (np.asarray(u.vertices)==np.asarray(v.vertices)).all()
                assert (np.asarray(u.triangles)==np.asarray(v.triangles)).all()
 
        # import ipdb;ipdb.set_trace()
        assert (mesh_rgb.vertices==mesh_normal.vertices)
        dtype = [  
            ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),  
            ('red', 'f8'), ('green', 'f8'), ('blue', 'f8'),  
            ('normal_x', 'f8'), ('normal_y', 'f8'), ('normal_z', 'f8'),    
            ('diffuse_r', 'f8'), ('diffuse_g', 'f8'), ('diffuse_b', 'f8'),
            ('albedo_r', 'f8'), ('albedo_g', 'f8'), ('albedo_b', 'f8'),
            ('metallic_0', 'f8'),
            ('roughness_0', 'f8'),
        ]  
        data = np.concatenate([
            np.asarray(mesh_rgb.vertices).astype(np.float64),
            np.asarray(mesh_rgb.vertex_colors).astype(np.float64),
            np.asarray(mesh_normal.vertex_colors).astype(np.float64)*2.-1.,
            np.asarray(mesh_diffuse.vertex_colors).astype(np.float64),
            np.asarray(mesh_basecolor.vertex_colors).astype(np.float64),
            np.asarray(mesh_metallic.vertex_colors).astype(np.float64)[:,:1],
            np.asarray(mesh_roughness.vertex_colors).astype(np.float64)[:,:1]
        ],axis=-1)
        vertices = data.view(dtype).reshape(-1)
        vertex_element = PlyElement.describe(vertices, 'vertex')

        triangles = np.asarray(mesh_rgb.triangles)
        face_element = PlyElement.describe(  
            np.array([(list(t),) for t in triangles],   
                    dtype=[('vertex_indices', 'i4', (3,))]),
            'face'  
        )
        ply_data = PlyData([vertex_element, face_element])
        ply_data.write(("material.ply" if save_path is None else save_path)) # for read

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    def visualize_depth(self,depth, near=0.2, far=13):
        import matplotlib
        if len(depth.shape) >= 3:
            depth = depth[0]
        depth = depth.detach().cpu().numpy()
        colormap = matplotlib.colormaps['turbo']
        curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
        eps = np.finfo(np.float32).eps
        near = near if near else depth.min()
        far = far if far else depth.max()
        near -= eps
        far += eps
        near, far, depth = [curve_fn(x) for x in [near, far, depth]]
        depth = np.nan_to_num(
            np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
        vis = colormap(depth)[:, :, :3]

        out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
        return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.visualize_depth(self.depthmaps[idx]).permute(1,2,0).cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))

        # render_path = os.path.join(path, "renders")
        # gts_path = os.path.join(path, "gt")
        # vis_path = os.path.join(path, "vis")
        # normal_path = os.path.join(path, "normal")
        # depth_path = os.path.join(path, "depth")
        # distance_path = os.path.join(path, "distance")
        # depth_normal_path = os.path.join(path, "depth_normal")
        # os.makedirs(render_path, exist_ok=True)
        # os.makedirs(vis_path, exist_ok=True)
        # os.makedirs(gts_path, exist_ok=True)
        # os.makedirs(normal_path, exist_ok=True)
        # os.makedirs(depth_path, exist_ok=True)
        # os.makedirs(distance_path, exist_ok=True)
        # os.makedirs(depth_normal_path, exist_ok=True)
        # # import ipdb;ipdb.set_trace()
        # for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
        #     gt = viewpoint_cam.original_image[0:3, :, :]
        #     save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        #     save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        #     # save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
        #     # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(normal_path, 'normal_{0:05d}'.format(idx) + ".png"))
        #     save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(depth_normal_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))

        #     normal = ((self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        #     cv2.imwrite(os.path.join(normal_path, 'normal_{0:05d}'.format(idx) + ".png"),normal)
        #     depth=self.depthmaps[idx]
        #     # distance=self.distance[idx]
        #     depth = depth.squeeze().detach().cpu().numpy()
        #     # distance = distance.squeeze().detach().cpu().numpy()
        #     depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        #     depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        #     depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        #     # distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
        #     # distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
        #     # distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)

        #     cv2.imwrite(os.path.join(depth_path, 'depth_{0:05d}'.format(idx) + ".png"),depth_color)
        #     # cv2.imwrite(os.path.join(distance_path, 'distance_{0:05d}'.format(idx) + ".png"),distance_color)

        #     visual_lst = self.all_maps[idx]
        #     grid = torch.stack(visual_lst,dim=0)
        #     grid = make_grid(grid,nrow=4)
        #     scale = grid.shape[-2] / 3600
        #     grid = torch.nn.functional.interpolate(
        #         grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale))
        #     )[0]
        #     save_image(grid, os.path.join(vis_path, 'all_map_{0:05d}'.format(idx) + ".png"))