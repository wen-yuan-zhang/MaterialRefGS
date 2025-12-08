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

import os
import torch
import open3d as o3d
import random
from random import randint
from utils.loss_utils import calculate_loss, l1_loss, get_img_grad_weight, lncc
# from gaussian_renderer import render_surfel, render_initial, render_volume, network_gui
from gaussian_renderer import render_initial,render_volume,render_surfel
from gaussian_renderer.envgs_renderer import render_surfel2
# from gaussian_renderer.envgs_renderer import render_surfel, render_surfel_with_envgs, render_surfel_with_envgs_sep,render_surfel2

from gaussian_renderer.optix_utils import HardwareRendering
import sys
from scene import Scene, GaussianModel
from scene.env_gaussian_model3 import EnvGaussianModel
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.image_utils import visualize_depth,visualize_depth2,visualize_d_mask,dilated_edges_imgs
from utils.graphics_utils import linear_to_srgb
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.graphics_utils import patch_offsets, patch_warp
from scene.cameras import Camera
from utils.camera_utils import gen_virtul_cam
from glob import glob
from collections import defaultdict
from typing import List, Optional
import imageio.v2 as imageio
import cv2
from joblib import Parallel,delayed
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

ROOT = os.getcwd().split("/")[1]

def load_by_method(method="GeoWizard",scan_id=None, args=None):
    if method == "GeoWizard":
        prior_rt = f"{args.geowizard_path}/dtu_prior/{scan_id}/normal_colored"
        glob_regx = "*_pred_colored.png"
        posix = "_pred_colored.png"
    elif method == "Metric3D":
        prior_rt = f"{args.metric3d_path}/{scan_id}/normal"
        glob_regx = "*.png"
        posix = ".png"
    elif method == "IDArb":
        prior_rt = f"{args.idarb_path}/{scan_id}/normal/"
        glob_regx = "*.png"
        posix = ".png"
    return prior_rt, glob_regx, posix


def load_normal_prior(dataset, scan_id:str, r=1, method="GeoWizard", args=None):
    if r<0:r=1
    prior_rt = args.idarb_path
    prior_rt2, glob_regx, posix = load_by_method(method="Metric3D", scan_id=scan_id, args=args)

    def glob_data(data_dir):
        data_paths = []
        data_paths.extend(glob(data_dir))
        data_paths = sorted(data_paths)
        return data_paths
    normal_paths = glob_data(os.path.join('{0}'.format(os.path.join(prior_rt2)), glob_regx))
    # print(normal_paths)
    # normal_paths = glob_data(os.path.join("/data1/tjm/code/PGSR2/output_dtu/dtu_scan110_2/test/train/ours_30000/renders_normal", "*.png"))
    normal_images = {}
    print("Loading Normal Maps...")
    for npath in normal_paths:
        normal = imageio.imread(npath)[...,:3]
        h,w = normal.shape[0],normal.shape[1]
        w2,h2 = int((w+0.5*r)//r),int((h+0.5*r)//r)
        normal = cv2.resize(normal,(w2,h2)).reshape(-1,3)
        normal = (normal/255.)*2. - 1
        # normal[:,1:] *= -1
        # normal *= -1
        img_name = os.path.basename(npath).replace(posix,"")
        normal_images.update({
            img_name: torch.from_numpy(normal).float()
        })
    print("Loading Mask Maps...")
    # import ipdb;ipdb.set_trace()
    def get_mask_dir(dataset_path):
        if "scan" in dataset_path:
            return os.path.join(dataset.source_path,"images")
        if "refnerf" in dataset_path:
            return os.path.join(dataset.source_path,"train")
        if "GlossySynthetic_blender" in dataset_path:
            return os.path.join(dataset.source_path,"rgb")
        if "refreal" in dataset_path:
            return f"{args.idarb_path}/{scan_id}/mask"

    def refnerf_maskpathfilter(paths):
        import re
        pattern = re.compile(r'^r_\d+\.png$')
        if "ball" in paths[0]:
            filtered_paths = [f for f in paths if "alpha" in f]
        else:
            filtered_paths = [f for f in paths if pattern.match(os.path.basename(f))]
        return filtered_paths
    mask_dir =  get_mask_dir(dataset.source_path)#os.path.join(dataset.source_path,"images") #TODO for refreal and mipnerf360, we don't  have a mask
    print(mask_dir)
    mask_paths = glob_data(os.path.join(mask_dir,"*.png"))
    if "refnerf" in mask_dir:
        mask_paths = refnerf_maskpathfilter(mask_paths)
    mask_images = {}
    for mpath in mask_paths:
        img = imageio.imread(mpath)
        if len(img.shape) == 2:
            img = img[...,None]
        h,w = img.shape[0],img.shape[1]
        mask = (img[...,-1] > 128).astype(np.float32)
        w2,h2 = int((w+0.5*r)//r),int((h+0.5*r)//r)
        mask = cv2.resize(mask,(w2,h2)).reshape(-1,1)

        img_name = os.path.basename(mpath).replace(".png","").replace(".jpg","").replace("_alpha","")
        mask_images.update(
            {img_name: torch.from_numpy(mask).float()}
        )

    print("Loading Metallic Maps...")
    # metal
    mtl_images = {}
    mtl_paths = glob_data(os.path.join('{0}'.format(os.path.join(prior_rt,scan_id,"mtl")), "*.png"))
    try:
        for npath in mtl_paths:
            mtl = imageio.imread(npath)
            h,w = mtl.shape[0],mtl.shape[1]
            mtl = cv2.resize(mtl,(w+0.5*r)//r,(h+0.5*r)//r)
            mtl = mtl/255.
            img_name = os.path.basename(npath)[:-4]
            
            mtl_images.update({
                img_name: torch.from_numpy(mtl).float().permute((2,0,1))[:1,...]
            })
    except:
        pass
    
    print("Loading Roughness Maps...")
    rgh_images = {}
    rgh_paths = glob_data(os.path.join('{0}'.format(os.path.join(prior_rt,scan_id,"rgh")), "*.png"))
    try:
        for npath in rgh_paths:
            rgh = imageio.imread(npath)
            h,w = rgh.shape[0],rgh.shape[1]
            rgh = cv2.resize(rgh,(w+0.5*r)//r,(h+0.5*r)//r)
            rgh = rgh/255.
            img_name = os.path.basename(npath)[:-4]
            
            rgh_images.update({
                img_name: torch.from_numpy(rgh).float().permute((2,0,1))[:1,...]
            })
    except:
        pass

    print("Loading Albeldo Maps...")
    albedo_images = {}
    albedo_paths = glob_data(os.path.join('{0}'.format(os.path.join(prior_rt,scan_id,"albeldo")), "*.png"))
    try:
        for npath in albedo_paths:
            albedo = imageio.imread(npath)
            h,w = albedo.shape[0],albedo.shape[1]
            albedo = cv2.resize(albedo,(w//r,h//r))
            albedo = albedo/255.
            img_name = os.path.basename(npath)[:-4]
            
            albedo_images.update({
                img_name: torch.from_numpy(albedo).float().permute((2,0,1))
            })
    except:
        pass

    ## TODO add ref_score_images loading
    print("Loading Ref Score Maps...")
    ref_score_images = {}
    ref_score_paths = glob_data(f"{args.ref_score_path}/"+"*.png")
    print(f"Finding {len(ref_score_paths)} ref score maps...")
    try:
        for npath in ref_score_paths:
            ref_score = imageio.imread(npath)[...,-1:]
            ref_score = (ref_score > 128)
            img_name = os.path.basename(npath)[:-4]
            # if img_name not in mask_images.keys():
                # continue
            # ref_score_images.update({img_name: np.zeros_like(ref_score)})
            # ref_score = np.logical_and(mask_images[img_name].reshape(-1).numpy(),ref_score.reshape(-1)).reshape(ref_score.shape)
            ref_score_images.update({img_name: torch.from_numpy(ref_score).bool().cuda().permute(2,0,1)})
    except:
        pass

    print("Finised Loading Prior Maps!!!")
    return mask_images, normal_images, mtl_images, rgh_images, albedo_images, ref_score_images

def mono_normal_loss(viewpoint_cam, surf_normal, rend_normal, mask_images, normal_images, gamma, iteration, ref_mask=None, HSV_mask=None):
    
    def get_normal_loss(normal_pred, normal_gt, mask=None):       
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        if mask is None:
            l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
            cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
            return l1, cos
        else:
            l1 = ((torch.abs(normal_pred - normal_gt)*mask).sum(dim=-1)).sum()/(mask.sum())
            cos = ((1. - torch.sum(normal_pred * normal_gt, dim=-1))*mask.squeeze()).sum()/(mask.sum())
            return l1, cos
    
    rot = viewpoint_cam.R.T.float().cuda()
    rot_surf_normal_map = (rot @ surf_normal.reshape(3,-1)).permute(1,0)    # (H*W,3)
    rot_rend_normal_map = (rot @ rend_normal.reshape(3,-1)).permute(1,0)

    img_name = viewpoint_cam.image_name
    mask = mask_images[img_name].cuda() if mask_images is not None else None
    # if ref_mask is not None:
    #     if mask is None:
    #         mask = ref_mask
    #     else:
    #         mask = ref_mask * mask
    # if HSV_mask is not None:
    #     mask = HSV_mask * mask
    l1_normal, cos_normal = get_normal_loss(rot_surf_normal_map, normal_images[img_name].cuda(),mask)
    l1_normal2, cos_normal2 = get_normal_loss(rot_rend_normal_map, normal_images[img_name].cuda(),mask)

    
    with torch.no_grad():
        _,H,W = surf_normal.shape
        # H,W = 581,777
        if iteration % 3000 == 0:
            os.makedirs("debug",exist_ok=True)
            mask = mask.reshape((H,W,1)).repeat((1,1,3))
            mask = (mask * 255).detach().cpu().numpy().astype(np.uint8)
            if HSV_mask is not None:
                HSV_mask = HSV_mask.reshape((H,W,1)).repeat((1,1,3))
                HSV_mask = (HSV_mask * 255).detach().cpu().numpy().astype(np.uint8)
            rot_surf_normal_map = rot_surf_normal_map.reshape((H,W,3))
            rot_rend_normal_map = rot_rend_normal_map.reshape((H,W,3))
            rot_surf_normal_map = (((rot_surf_normal_map + 1) * 0.5) * 255).detach().cpu().numpy().astype(np.uint8)
            rot_rend_normal_map = (((rot_rend_normal_map + 1) * 0.5) * 255).detach().cpu().numpy().astype(np.uint8)
            normal = normal_images[img_name].reshape((H,W,3))
            normal = (((normal + 1) * 0.5) * 255).detach().cpu().numpy().astype(np.uint8)

            imageio.imwrite(f"./debug/{iteration}_mask.png",mask)
            imageio.imwrite(f"./debug/{iteration}_rot_surf_normal_map_{iteration}.png",rot_surf_normal_map)
            imageio.imwrite(f"./debug/{iteration}_rot_rend_normal_map_{iteration}.png",rot_rend_normal_map)
            imageio.imwrite(f"./debug/{iteration}_normal_{iteration}.png",normal)
            if HSV_mask is not None:
                imageio.imwrite(f"./debug/{iteration}_HSV_mask.png",HSV_mask)


    return l1_normal, cos_normal, l1_normal2, cos_normal2
    return 0.01 * (l1_normal + cos_normal + l1_normal2 + cos_normal2)


def cal_material_loss(viewpoint_cam,roughness_map,refl_strength_map,albedo_map,mask_images,mtl_images,rgh_images,albedo_images):
    img_name = viewpoint_cam.image_name
    H,W=refl_strength_map.shape[1],refl_strength_map.shape[2]
    mask = mask_images[img_name].cuda().reshape(-1) if mask_images is not None else None
    mask = mask.reshape((1,H,W))
    roughness_loss = (torch.abs(roughness_map-rgh_images[img_name].cuda()) * mask).sum() / (mask.sum() + 1e-8)
    ref_loss = (torch.abs(refl_strength_map-mtl_images[img_name].cuda()) * mask).sum() / (mask.sum() + 1e-8)
    albedo_loss = (torch.abs(albedo_map-albedo_images[img_name].cuda()) * mask).sum() / (mask.sum() + 1e-8)
    return roughness_loss,ref_loss,albedo_loss


def get_HSV_mask(gt_image,mask_images,viewpoint_cam):
    gt_image_np = (gt_image.cpu().permute((1,2,0)) * 255).numpy().astype(np.uint8)
    hsv_image = cv2.cvtColor(gt_image_np, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)
    # V=(255-V)
    V[V<50]=0
    V = torch.from_numpy(V).cuda()
    V = (V/255).reshape(-1,1) * mask_images[viewpoint_cam.image_name].cuda()
    V = torch.exp(V) - 1
    return V

def debug():
    pass #TODO add warp debug information

def rgb2gray(inputs):
    _,a,b=inputs.shape
    gray_image = (0.299 * inputs[0] + 0.587 * inputs[1] + 0.114 * inputs[2]).reshape(1,a,b)
    return gray_image

def get_consistency_loss(
    ref_view_maps,
    nst_view_maps, # TODO add learned material maps
    weights,
    ncc_weight,
    total_patch_size
):
    pixels_patch,gt_image_gray,albeldo,rgh,mtl = ref_view_maps
    pixels_patch_nst,gt_image_gray_nst,albeldo_nst,rgh_nst,mtl_nst = nst_view_maps


    # F.grid_smaple(input,grid) input:(N,C,H_in,W_in) grid:(N,H_out,W_out,2), values of grid lie within[-1,1]
    ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)
    ref_mtl_val = F.grid_sample(rgh.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_mtl_val = ref_mtl_val.reshape(-1, total_patch_size)
    ref_rgh_val = F.grid_sample(mtl.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_rgh_val = ref_rgh_val.reshape(-1, total_patch_size)
    ref_albeldo_val = F.grid_sample(albeldo.unsqueeze(0), pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_albeldo_val = ref_albeldo_val.reshape(3,-1,total_patch_size)
    ref_albeldo_gray_val = rgb2gray(ref_albeldo_val)


    sampled_gray_val = F.grid_sample(gt_image_gray_nst[None], pixels_patch_nst.reshape(1, -1, 1, 2), align_corners=True)
    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
    nst_mtl_val = F.grid_sample(rgh_nst.unsqueeze(1), pixels_patch_nst.reshape(1, -1, 1, 2), align_corners=True)
    nst_mtl_val = nst_mtl_val.reshape(-1, total_patch_size)
    nst_rgh_val = F.grid_sample(mtl_nst.unsqueeze(1), pixels_patch_nst.reshape(1, -1, 1, 2), align_corners=True)
    nst_rgh_val = nst_rgh_val.reshape(-1, total_patch_size)
    nst_albeldo_val = F.grid_sample(albeldo_nst.unsqueeze(0), pixels_patch_nst.reshape(1, -1, 1, 2), align_corners=True)
    nst_albeldo_val = nst_albeldo_val.reshape(3,-1,total_patch_size)
    nst_albeldo_gray_val = rgb2gray(nst_albeldo_val)

    mtl_thr = 0.9
    rgh_thr = 0.1
    
    # NOTE transform albeldo to gray?
    mask1 = torch.bitwise_or((ref_mtl_val>mtl_thr),(ref_mtl_val<rgh_thr))
    mask2 = torch.bitwise_or((nst_mtl_val>mtl_thr),(nst_rgh_val<rgh_thr))
    mask = torch.bitwise_and(mask1,mask2)
    ref_val = mask * ref_albeldo_gray_val + (~mask) * ref_gray_val
    nst_val = mask * nst_albeldo_gray_val + (~mask) * sampled_gray_val
    # 或者只mask掉高亮部分,并不用高亮部分作为warp
    # ref_val = ref_gray_val
    # nst_val = sampled_gray_val

    ncc, ncc_mask = lncc(ref_val, nst_val)
    mask = ncc_mask.reshape(-1)
    ncc = ncc.reshape(-1) * weights
    ncc = ncc[mask].squeeze()
    ncc_loss = None
    if mask.sum() > 0:
        ncc_loss = ncc_weight * ncc.mean()
    return ncc_loss

def get_consistency_loss3(
    render_pkg,render_pkg_nst,
    ref_gray_val,sampled_gray_val,
    pixels_patch,pixels_patch_nst,total_patch_size,weights,ncc_weight
):
    specular = render_pkg["specular_map"]
    specular_nst = render_pkg_nst["specular_map"]
    with torch.no_grad(): # NOTE: stop gradient for weight caculation
        # F.grid_smaple(input,grid) input:(N,C,H_in,W_in) grid:(N,H_out,W_out,2), values of grid lie within[-1,1]
        specular_map_val = F.grid_sample(specular.unsqueeze(0), pixels_patch.reshape(1, -1, 1, 2), align_corners=True)
        specular_map_val = specular_map_val.reshape(-1, 3, total_patch_size)
        specular_nst_val = F.grid_sample(specular_nst.unsqueeze(0), pixels_patch_nst.reshape(1, -1, 1, 2), align_corners=True)
        specular_nst_val = specular_nst_val.reshape(-1, 3, total_patch_size)
        specular_map_gray_val = rgb2gray(specular_map_val.permute((1,0,2))).reshape(-1,total_patch_size)
        specular_nst_gray_val = rgb2gray(specular_nst_val.permute((1,0,2))).reshape(-1,total_patch_size)
        ref_ratio = torch.min(specular_map_gray_val/(1e-5+ref_gray_val),torch.tensor([1.]).to(specular_map_gray_val.device)).mean(-1)
        nst_ratio = torch.min(specular_nst_gray_val/(1e-5+sampled_gray_val),torch.tensor([1.]).to(specular_map_gray_val.device)).mean(-1)
        ref_weight = 1. - torch.log2((ref_ratio+nst_ratio)/2.+1.) # (N)

    weights *= ref_weight

    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
    mask = ncc_mask.reshape(-1)
    ncc = ncc.reshape(-1) * weights
    ncc = ncc[mask].squeeze()
    ncc_loss = None
    if mask.sum() > 0:
        ncc_loss = ncc_weight * ncc.mean()
    return ncc_loss

def get_consistency_loss2(
        render_pkg, 
        render_pkg_nst,
        ref_gray_val:torch.Tensor,
        sampled_gray_val:torch.Tensor,
        pixels_patch,
        pixels_patch_nst,
        total_patch_size,
        weights,
        ncc_weight,ncc_scale #NOTE?
    ):
    # import ipdb;ipdb.set_trace()
    ref_map = render_pkg["refl_strength_map"] # (1,H,W)
    ref_map_nst = render_pkg_nst["refl_strength_map"] # TODO 添加遮挡map? 
    
    with torch.no_grad(): # NOTE: stop gradient for weight caculation
        # F.grid_smaple(input,grid) input:(N,C,H_in,W_in) grid:(N,H_out,W_out,2), values of grid lie within[-1,1]
        # 这里其实不做scale,只是修改一下patch_size和material map size?
        ref_map_val = F.grid_sample(ref_map.unsqueeze(1), pixels_patch.reshape(1, -1, 1, 2), align_corners=True)
        ref_map_val = ref_map_val.reshape(-1, total_patch_size).mean(-1)
        ref_map_nst_val = F.grid_sample(ref_map_nst.unsqueeze(1), pixels_patch_nst.reshape(1, -1, 1, 2), align_corners=True)
        ref_map_nst_val = ref_map_nst_val.reshape(-1, total_patch_size).mean(-1)

        ref_weight = 1.0-(ref_map_val+ref_map_nst_val)/2
        # ref_weight = 1.-torch.log2((ref_map_val+ref_map_nst_val)/2.+1.) # (N)
        ref_weight[ref_weight<0.9] = 0
        # ref_weight_thr = 0.1
        valid_indices = ((ref_map_val+ref_map_nst_val)<0.4)
    # weights *= ref_weight

    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
    mask = ncc_mask.reshape(-1)
    mask = torch.bitwise_and(mask,valid_indices)
    ncc = ncc.reshape(-1) * weights
    ncc = ncc[mask].squeeze()
    ncc_loss = None
    if mask.sum() > 0:
        ncc_loss = ncc_weight * ncc.mean()
    return ncc_loss,ref_weight.detach()

def debug_distance(distance):
    pass
    # distance = distance.detach().cpu().numpy()
    # distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
    # distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
    # distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
    # os.makedirs("debug_distance",exist_ok=True)
    # cv2.imwrite(distance_color,"./debug_distance/debug.png")

def visual_refweight(ref_weight,pixels,H,W):
    pixels = pixels.reshape(-1,2).detach().cpu()
    ref_weight_map = torch.zeros((H,W))
    ref_weight_map[pixels[:,1].long(),pixels[:,0].long()]=ref_weight.detach().cpu()
    return ref_weight_map

def calc_warp_loss(
        viewpoint_cam:Camera,
        scene:Scene,
        opt:OptimizationParams, # TODO add parameters for warp
        gaussians:GaussianModel,
        dataset:ModelParams,
        pipe:PipelineParams,
        render,
        render_pkg,
        albeldo_images,
        mtl_images,
        rgh_images,
        mask_images,
        # ird_images,
        iteration:int,
        # app_model,
        debug_path,
        bg,
        use_metallic_warp=False,
        use_roughness_warp=False
):
    gt_image, gt_image_gray = viewpoint_cam.get_image()
    image_name = viewpoint_cam.image_name
    dilate_size = 7 #opt.dilate_size
    edge_aware = opt.edge_aware_in_warp
    # albeldo,mtl,rgh = albeldo_images[image_name].cuda(),mtl_images[image_name].cuda(),rgh_images[image_name].cuda() 
    # (3,H,W)
    geo_loss = ncc_loss = base_color_loss = base_color_loss_map = None
    metallic_warp_loss = roughness_warp_loss = None
    original_weight = None
    # import ipdb;ipdb.set_trace()

    with torch.no_grad():
        ref_normal = render_pkg['rend_normal']
        edges_mask = dilated_edges_imgs(ref_normal, dilate_size=dilate_size).reshape(-1).bool().cuda()
        edges_mask = ~edges_mask
        if not edge_aware:
            edges_mask = torch.ones_like(edges_mask).bool().cuda()

    # viewpoint_cam
    nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
    use_virtul_cam = False
    if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
        nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
        use_virtul_cam = True
    if nearest_cam is not None:

        # nearest view material map
        image_name_nearest= nearest_cam.image_name
        # albeldo_nst,mtl_nst,rgh_nst = albeldo_images[image_name_nearest].cuda(),mtl_images[image_name_nearest].cuda(),rgh_images[image_name_nearest].cuda() 

        patch_size = opt.multi_view_patch_size
        sample_num = opt.multi_view_sample_num
        pixel_noise_th = opt.multi_view_pixel_noise_th
        total_patch_size = (patch_size * 2 + 1) ** 2
        ncc_weight = opt.multi_view_ncc_weight
        geo_weight = opt.multi_view_geo_weight
        metallic_weight = opt.metallic_warp_weight
        roughness_weight = opt.roughness_warp_weight 
        ## compute geometry consistency mask and loss
        H, W = render_pkg['surf_depth'].squeeze().shape # TODO 需要使用相同的depth
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['surf_depth'].device)

        # nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
        #                             return_plane=True, return_depth_normal=False) # TODO modify 
        # import ipdb;ipdb.set_trace()
        nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, srgb=opt.srgb, opt=opt, wo_render_img=False)

        pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['surf_depth']) # pts in world view
        pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
        map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['surf_depth'], pts_in_nearest_cam)
        # map_z: (H,W) src view pts在 ref view中的插值深度以及在ref view fov之外的点的mask

        pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
        pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
        R = nearest_cam.R.float().cuda()
        T = nearest_cam.T.float().cuda()
        pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2) # 用ref view depth还原的世界坐标的点
        pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
        pts_projections = torch.stack(
                    [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                    pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
        
        pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
        if not opt.wo_use_geo_occ_aware: # TODO add wo_use_geo_occ_aware
            d_mask = d_mask & (pixel_noise < pixel_noise_th)
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            weights[~d_mask] = 0
        else:
            d_mask = d_mask
            weights = torch.ones_like(pixel_noise)
            weights[~d_mask] = 0

        original_weight = weights.detach().clone()
        if d_mask.sum() > 0:
            geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
            # loss += geo_loss
            if use_virtul_cam is False:
                with torch.no_grad():
                    ## sample mask
                    d_mask = d_mask.reshape(-1) # (H*W)
                    valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask] # (没有被遮挡而且共有的)
                    if d_mask.sum() > sample_num:
                        index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                        valid_indices = valid_indices[index]

                    weights = weights.reshape(-1)[valid_indices]
                    ## sample ref frame patch
                    pixels = pixels.reshape(-1,2)[valid_indices] # (H*W,2) -> (N_sample,2)
                    offsets = patch_offsets(patch_size, pixels.device) # 以sampled pixel为中心生成patch
                    ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float() # (N_sample,Patch_size,2) # 这里对像素坐标进行了scale
                    
                    H, W = gt_image_gray.squeeze().shape
                    pixels_patch = ori_pixels_patch.clone()
                    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0 #用gt_image_gray作为warp对象
                    ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)
                    # F.grid_smaple(input,grid) input:(N,C,H_in,W_in) grid:(N,H_out,W_out,2), values of grid lie within[-1,1]

                    if iteration > 10000:
                        # with torch.no_grad():
                        #### Base Color Warp #Albeldo
                        rend_base_color = render_pkg["diffuse_map"]
                        # rend_base_color_gray = rgb2gray(rend_base_color)
                        base_color = F.grid_sample(rend_base_color.unsqueeze(0), \
                            pixels_patch.view(1,-1,1,2).detach(), align_corners=True) # (1,3,N*P,1)
                        base_color = base_color.reshape(3, -1, total_patch_size)
                        #### End Base Color Warp

                        if use_metallic_warp:
                            #### Metallic Warp
                            rend_metallic_map = render_pkg["refl_strength_map"]
                            metallic_map = F.grid_sample(rend_metallic_map.unsqueeze(0), \
                                pixels_patch.view(1,-1,1,2).detach(), align_corners=True) # (1,1,N*P,1)
                            metallic_map = metallic_map.reshape(1, -1, total_patch_size)
                            #### End Metallic Warp

                        if use_roughness_warp:
                            #### Roughness Warp
                            rend_roughness_map = render_pkg["roughness_map"]
                            roughness_map = F.grid_sample(rend_roughness_map.unsqueeze(0), \
                                pixels_patch.view(1,-1, 1, 2).detach(), align_corners=True) # (1,1,N*P,1)
                            roughness_map = roughness_map.reshape(1, -1, total_patch_size)
                            ### End Roughness Warp
                        
                                                

                    ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                    ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                ## compute Homography
                ref_local_n = ((render_pkg["rend_normal"].permute(1,2,0)) @ viewpoint_cam.world_view_transform[:3,:3]).reshape(-1,3) # TODO loal normal
                
                assert "rend_distance" in render_pkg.keys()
                if not "rend_distance" in render_pkg.keys():
                    # ref_local_d = render_pkg['rendered_distance'].squeeze()
                    rays_d = viewpoint_cam.get_rays()
                    # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3) # TODO local normal
                    ref_local_d = render_pkg['surf_depth'].view(-1) * ((ref_local_n * rays_d.reshape(-1,3)).sum(-1).abs())
                    ref_local_d = ref_local_d.reshape(*render_pkg['surf_depth'].shape)
                    debug_distance(ref_local_d)
                else:
                    # print("DEBUG")
                    ref_local_d = render_pkg["rend_distance"].squeeze()


                ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]
                ref_local_d = ref_local_d.reshape(-1)[valid_indices]

                H_ref_to_neareast = ref_to_neareast_r[None] - \
                    torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale) # 也就是说如果要对材质也做warp的话，材质也应该是和这个一样的尺寸?
                
                ## compute neareast frame patch
                grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                _, nearest_image_gray = nearest_cam.get_image()
                sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                if iteration > 10000:
                    # with torch.no_grad():
                    #### Base Color Warp
                    rend_base_color_nst = nearest_render_pkg["diffuse_map"]
                    # rend_base_color_gray_nst = rgb2gray(rend_base_color_nst)
                    base_color_nst = F.grid_sample(rend_base_color_nst.unsqueeze(0), grid.reshape(1,-1,1,2).detach(), align_corners=True)
                    base_color_nst = base_color_nst.reshape(3,-1,total_patch_size)
                    #### End Base Color Warp

                    if use_metallic_warp:
                        #### Metallic Warp
                        rend_metallic_map_nst = nearest_render_pkg["refl_strength_map"]
                        metallic_map_nst = F.grid_sample(rend_metallic_map_nst.unsqueeze(0), grid.reshape(1,-1,1,2).detach(), align_corners=True)
                        metallic_map_nst = metallic_map_nst.reshape(1,-1,total_patch_size)
                        #### End Metallic Warp
                    
                    if use_roughness_warp:
                        #### Roughness Warp
                        rend_roughness_map_nst = nearest_render_pkg["roughness_map"]
                        roughness_map_nst = F.grid_sample(rend_roughness_map_nst.unsqueeze(0), grid.reshape(1,-1,1,2).detach(), align_corners=True)
                        roughness_map_nst = roughness_map_nst.reshape(1,-1,total_patch_size)
                        ### End Roughness Warp

                    # # Ignore the background
                    valid_indices_basecolor = None
                    with torch.no_grad():
                        # mask = viewpoint_cam.gt_alpha_mask # (H,W)
                        H,W,_ = viewpoint_cam.HWK
                        mask = mask_images[viewpoint_cam.image_name].cuda().squeeze().reshape((1,1,H,W))
                        

                        ## TODO combine bg mask indices and edge mask indices
                        mask_val = F.grid_sample(mask, pixels_patch.view(1, -1, 1, 2).detach(), align_corners=True)
                        mask_val = mask_val.reshape(-1, total_patch_size)
                        bgoredge_mask = mask_val.min(-1)[0] > 0.99
                        bgoredge_mask = torch.bitwise_and(bgoredge_mask,edges_mask[valid_indices])
                        # base_color = base_color[:,valid_indices,:]
                        # base_color_nst = base_color_nst[:,valid_indices,:]
                        # weights_basecolor = weights[valid_indices]
                        # valid_indices_basecolor = valid_indices.clone().to(pixels.device)
                    # Ignore the background

                    # Ncc loss for base color
                    base_color_loss = ((torch.abs(base_color-base_color_nst).sum(0).mean(-1) * weights.detach())).mean()
                    def L(d,gamma=0.2,delta=5.):
                        mask = d<gamma
                        d[mask] = (d[mask]/gamma)**3 * gamma
                        d[~mask] = (d[~mask]+1./delta*(torch.exp(delta*(d[~mask]-gamma))-1.))
                        return d
                    def L2(d:torch.Tensor):
                        return -torch.log((1.-d).clamp(1e-6,1-1e-6))
                    # Ncc loss for metallic
                    opt.directional_rghmtl_warp_alignment = True
                    if use_metallic_warp:
                        # metallic_warp_loss = ((torch.abs(metallic_map-metallic_map_nst).sum(0).mean(-1) * weights.detach())).mean()
                        if opt.directional_rghmtl_warp_alignment:
                            with torch.no_grad():
                                metallic_map_max = torch.max(metallic_map_nst,metallic_map).detach()
                                # metallic_map_max_mask = metallic_map_max.mean(-1) > 0.5
                                # bgoredge_mask = torch.bitwise_and(metallic_map_max_mask.squeeze(),bgoredge_mask)
                                metallic_value_weight = metallic_map_max.sum(0).mean(-1)

                            metallic_warp_loss =  (metallic_value_weight * ((torch.abs(metallic_map-metallic_map_max).sum(0).mean(-1) * weights.detach())))[bgoredge_mask]#.mean()
                            metallic_warp_loss += (metallic_value_weight * ((torch.abs(metallic_map_nst-metallic_map_max).sum(0).mean(-1) * weights.detach())))[bgoredge_mask]#.mean()
                            metallic_warp_loss = L(metallic_warp_loss).mean()
                            # import ipdb;ipdb.set_trace()
                        else:
                            metallic_warp_loss = ((torch.abs(metallic_map-metallic_map_nst).sum(0).mean(-1) * weights.detach()))[bgoredge_mask].mean()
                    # Ncc loss for roughness
                    if use_roughness_warp:
                        # roughness_warp_loss = ((torch.abs(roughness_map-roughness_map_nst).sum(0).mean(-1) * weights.detach())).mean()
                        if opt.directional_rghmtl_warp_alignment:
                            with torch.no_grad():
                                roughness_map_min = torch.min(roughness_map_nst,roughness_map).detach()
                                # roughness_map_min_mask = roughness_map_min.mean(-1) < 0.5
                                # bgoredge_mask = torch.bitwise_and(roughness_map_min_mask.squeeze(),bgoredge_mask)
                                
                            roughness_warp_loss = ((torch.abs(roughness_map-roughness_map_min).sum(0).mean(-1) * weights.detach()))[bgoredge_mask]#.mean()
                            roughness_warp_loss += ((torch.abs(roughness_map_nst-roughness_map_min).sum(0).mean(-1) * weights.detach()))[bgoredge_mask]#.mean()
                            roughness_warp_loss = L(roughness_warp_loss).mean()
                        else:
                            roughness_warp_loss = ((torch.abs(roughness_map-roughness_map_nst).sum(0).mean(-1) * weights.detach()))[bgoredge_mask].mean()

                    def get_current_basecolor_warp_weight(iteration):
                        return .1
                        start,end = 10000, 20000
                        if iteration < 12000:
                            return 4
                        elif iteration >= 12000 and iteration <= end:
                            return 4-(iteration-12000)/(end-12000) * (4-1.5)
                        else:
                            return 1.5
                    def get_current_mtlrgh_warp_weight(iteration):
                        return .5
                        start,end = 10000, 20000
                        if iteration < 12000:
                            return 4
                        elif iteration >= 12000 and iteration < end:
                            return 4-(iteration-12000)/(end-12000) * (4-1)
                        else:
                            return 1
                    base_color_loss = get_current_basecolor_warp_weight(iteration) * ncc_weight * base_color_loss
                    if use_metallic_warp:
                        metallic_warp_loss = get_current_mtlrgh_warp_weight(iteration) * metallic_weight * metallic_warp_loss
                    if use_roughness_warp:
                        roughness_warp_loss = get_current_mtlrgh_warp_weight(iteration) * roughness_weight * roughness_warp_loss
                    with torch.no_grad():
                        H,W=render_pkg['surf_depth'].squeeze().shape
                        base_color_loss_map = torch.abs(base_color-base_color_nst).sum(0).mean(-1) # (N)
                        try:
                            if valid_indices_basecolor is None:
                                base_color_loss_map = visual_refweight(base_color_loss_map,pixels,H,W)
                            else:
                                base_color_loss_map = visual_refweight(base_color_loss_map,pixels[valid_indices_basecolor],H,W)
                        except:
                            base_color_loss_map = None
                ## compute loss
                # ncc_loss = get_consistency_loss(
                #     (pixels_patch,gt_image_gray,albeldo,rgh,mtl),
                #     (grid,nearest_image_gray,albeldo_nst,rgh_nst,mtl_nst),
                #     weights,ncc_weight,total_patch_size
                # )
                # ncc_loss,ref_weight = get_consistency_loss2(
                #     render_pkg,nearest_render_pkg,ref_gray_val,sampled_gray_val,
                #     pixels_patch,grid,total_patch_size,weights,ncc_weight,viewpoint_cam.ncc_scale
                # )
                # ncc_loss = get_consistency_loss3(
                #     render_pkg,nearest_render_pkg,ref_gray_val,sampled_gray_val,
                #     pixels_patch,grid,total_patch_size,weights,ncc_weight
                # )
                # ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                # mask = ncc_mask.reshape(-1)
                # ncc = ncc.reshape(-1) * weights
                # ncc = ncc[mask].squeeze()
                # if mask.sum() > 0:
                #     ncc_loss = ncc_weight * ncc.mean()
    # if iteration % 5 != 0:
    #     base_color_loss = None
    # if iteration % 10 != 0:
    #     roughness_warp_loss = None
    # if iteration % 10 != 0:
    #     metallic_warp_loss = None

    H,W=render_pkg['surf_depth'].squeeze().shape
    return None, None, base_color_loss, metallic_warp_loss, roughness_warp_loss ,original_weight, None,None# base_color_loss_map

def get_sam2_mask():
    pass

def get_nearest_cam_lst(viewpoint_cam:Camera,viewpoint_stack:List[Camera]) -> List[Camera]:
    return [viewpoint_stack[_] for _ in viewpoint_cam.nearest_id]

def get_multi_view_neighbor(scene:Scene):
    nearest_neighbor_dicts = defaultdict()
    multi_view_num = 20
    multi_view_max_angle = 90
    multi_view_min_angle = 5
    multi_view_min_dis = 0.1
    multi_view_max_dis = 1.5
        
    print("computing nearest_id")
    world_view_transforms = []
    camera_centers = []
    center_rays = []
    for id, cur_cam in enumerate(scene.getTrainCameras().copy()):
        world_view_transforms.append(cur_cam.world_view_transform)
        camera_centers.append(cur_cam.camera_center)
        R = cur_cam.R.float().cuda()
        T = cur_cam.T.float().cuda()
        center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
        center_ray = center_ray@R.transpose(-1,-2)
        center_rays.append(center_ray)
    world_view_transforms = torch.stack(world_view_transforms)
    camera_centers = torch.stack(camera_centers, dim=0)
    center_rays = torch.stack(center_rays, dim=0)
    center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
    diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy()
    tmp = torch.sum(center_rays[:,None]*center_rays[None], dim=-1)
    angles = torch.arccos(tmp)*180/3.14159
    angles = angles.detach().cpu().numpy()
    for id, cur_cam in enumerate(scene.getTrainCameras().copy()):
        sorted_indices = np.lexsort((angles[id], diss[id]))
        # sorted_indices = np.lexsort((diss[id], angles[id]))
        mask = (angles[id][sorted_indices] < multi_view_max_angle) & \
            (diss[id][sorted_indices] > multi_view_min_dis) & \
            (diss[id][sorted_indices] < multi_view_max_dis) & \
            (angles[id][sorted_indices] > multi_view_min_angle)
        sorted_indices = sorted_indices[mask]
        multi_view_num = min(multi_view_num, len(sorted_indices))
        nearest_neighbor_dicts[cur_cam.image_name] = [
            (index,scene.getTrainCameras()[index].image_name) for index in sorted_indices[:multi_view_num]
        ]
    return nearest_neighbor_dicts


@torch.no_grad()
def calc_ref_score(
    scene:Scene,
    opt:OptimizationParams, # TODO add parameters for warp
    gaussians:GaussianModel,
    dataset:ModelParams,
    pipe:PipelineParams,
    albeldo_images,
    mtl_images,
    rgh_images,
    mask_images,
    iteration,
    bg
):     
    # import ipdb;ipdb.set_trace()
    patch_size = 4
    dilate_size = 2
    edges_aware = False
    total_patch_size = (2 * patch_size + 1) ** 2
    normal_map_cached = defaultdict()
    depth_map_cached = defaultdict()
    distance_map_cached = defaultdict()

    viewpoint_stack:List[Camera] = scene.getTrainCameras().copy()
    for viewpoint_cam in tqdm(viewpoint_stack,desc="Rendering...",total=len(viewpoint_stack)):
        image_name = viewpoint_cam.image_name
        render_pkg= render_surfel(viewpoint_cam,gaussians,pipe,bg,srgb=opt.srgb,opt=opt)
        normal_map_cached[image_name] = render_pkg["rend_normal"].detach().cpu()
        depth_map_cached[image_name] = render_pkg["surf_depth"].detach().cpu()
        distance_map_cached[image_name] = render_pkg["rend_distance"].detach().cpu()

    ref_score_images = defaultdict()
    import ipdb;ipdb.set_trace()    
    viewpoint_stack:List[Camera] = scene.getTrainCameras().copy()
    nearest_neighbor_dicts = get_multi_view_neighbor(scene)
    for viewpoint_cam in tqdm(viewpoint_stack,desc="Computing Reflection Score...",total=len(viewpoint_stack)):
        ref_image_name = viewpoint_cam.image_name
        ref_normal:torch.Tensor = normal_map_cached[ref_image_name].cuda()
        ref_depth:torch.Tensor = depth_map_cached[ref_image_name].cuda()
        ref_distance:torch.Tensor = distance_map_cached[ref_image_name].cuda()
        H,W = ref_depth.squeeze().shape

        # nearest_cam_lst = get_nearest_cam_lst(viewpoint_cam,viewpoint_stack)
        nearest_cam_lst = [viewpoint_stack[idx] for (idx,_) in nearest_neighbor_dicts[ref_image_name]]
        ## 采点
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        pixels = torch.stack([ix, iy], dim=-1).float().to(ref_depth.device)

        ## 遮挡检测
        pts = gaussians.get_points_from_depth(viewpoint_cam, ref_depth)
        d_mask_lst = []
        sampled_rgb_lst = []
        anchored_rgb = None
        for nearest_cam in nearest_cam_lst:
            pts_in_nearest_cam = pts.clone() @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
            depth_z_in_nearest_cam = pts_in_nearest_cam[:,2:3]
            map_z, d_mask = gaussians.get_points_depth_in_depth_map(
                nearest_cam, depth_map_cached[nearest_cam.image_name].cuda(), pts_in_nearest_cam
            )
            pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
            pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
            R = nearest_cam.R.float().cuda()
            T = nearest_cam.T.float().cuda()
            pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
            pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
            pts_projections = torch.stack(
                        [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                        pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
            
            pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
            d_mask = d_mask & (pixel_noise < opt.multi_view_pixel_noise_th)
            
            # patch warp
            d_mask = d_mask.reshape(-1) # (H*W)
            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask] 
            pixels_this_neighbor = pixels.clone().reshape(-1,2)[valid_indices] #(M,2) range [0,H]x[0,W]
            offsets = patch_offsets(patch_size, pixels_this_neighbor.device)
            ori_pixels_patch = pixels_this_neighbor.reshape(-1, 1, 2) + offsets.float()

            if anchored_rgb is None:
                ref_pixels_patch = pixels.clone().reshape(-1,1,2) + offsets.float()
                ref_pixels_patch[:, :, 0] = 2 * ref_pixels_patch[:, :, 0] / (W - 1) - 1.0
                ref_pixels_patch[:, :, 1] = 2 * ref_pixels_patch[:, :, 1] / (H - 1) - 1.0
                ref_image_rgb,_ = viewpoint_cam.get_image()
                
                anchored_rgb = F.grid_sample(
                    ref_image_rgb.unsqueeze(0), ref_pixels_patch.reshape(1,-1,1,2), align_corners=True
                )
                anchored_rgb = anchored_rgb.reshape(3,-1,total_patch_size)


            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

            ## compute Homography
            ref_local_n = ((ref_normal.permute(1,2,0)) @ viewpoint_cam.world_view_transform[:3,:3]).reshape(-1,3) # TODO local normal

            ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]
            ref_local_d = ref_distance.reshape(-1)[valid_indices]

            H_ref_to_neareast = ref_to_neareast_r[None] - \
                torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                            ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
            H_ref_to_neareast = torch.matmul(nearest_cam.get_k()[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
            H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k()

            grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
            grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
            grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
            image_rgb,_ = nearest_cam.get_image()
            
            sampled_rgb = F.grid_sample(
                image_rgb.unsqueeze(0), grid.reshape(1,-1,1,2), align_corners=True
            )
            sampled_rgb = sampled_rgb.reshape(3,-1,total_patch_size)
            
            
            d_mask_lst += [d_mask.detach().cpu()]
            sampled_rgb_map = torch.zeros((3,H,W,total_patch_size)).to(sampled_rgb.device)
            sampled_rgb_map[
                :,pixels_this_neighbor[:,1].long(),pixels_this_neighbor[:,0].long(),:
            ] = sampled_rgb # (3,-1,P^2)
            sampled_rgb_lst += [sampled_rgb_map.reshape((3,-1,total_patch_size))]
        
        all_d_mask = torch.stack(d_mask_lst).cuda() #(N,-1)
        
        all_sampled_rgb = torch.stack(sampled_rgb_lst).permute((0,2,3,1)).contiguous().cuda() #(N,-1,P^2,3)
        anchored_rgb = anchored_rgb.unsqueeze(0).permute((0,2,3,1)).contiguous().cuda() #(1,-1,P^2,3)
        
        ## 边缘检测
        # import ipdb;ipdb.set_trace()
        edges_mask = dilated_edges_imgs(ref_normal, dilate_size=dilate_size).reshape(H*W) # (0,1) (not edges, edges)
        # import ipdb;ipdb.set_trace()

        try:
            ## mean std
            RS_temp = 10. * torch.zeros(all_sampled_rgb.shape[1]).cuda()
            tot_sampled_rgb = torch.concat([all_sampled_rgb,anchored_rgb],dim=0)
            tot_d_mask = torch.concat([all_d_mask,torch.ones(1,all_d_mask.shape[-1]).bool().cuda()])
            mean_color = tot_sampled_rgb.sum(0,keepdim=True)/tot_d_mask.sum(0,keepdim=True)[:,:,None,None]
            std_color = (((tot_sampled_rgb-mean_color)**2)*(tot_d_mask.float()[...,None,None])).sum(0)/(tot_d_mask.sum(0)[:,None,None])**0.5
            RS_temp = std_color.sum(-1).mean(-1)

            save_dir = f"debug_refreal/{iteration}"
            os.makedirs(save_dir,exist_ok=True)
            RS_temp_vis = RS_temp.reshape((H,W))/RS_temp.max()
            save_image(RS_temp_vis,f"{save_dir}/{ref_image_name}_ref_score_std.png")
        except:
            pass
        ## mahalanobis_distance https://en.wikipedia.org/wiki/Mahalanobis_distance
        # def cov(i,j):
        #     val_inds = (all_d_mask.sum(0,keepdim=True)>0)
        #     mean_i = torch.zeros 
        #     pass

        try:
            RS_temp = 10. * torch.zeros(all_sampled_rgb.shape[1]).cuda()
            tot_sampled_rgb = torch.concat() 
        except:
            pass


        try:
            ## coefficient of variation https://en.wikipedia.org/wiki/Coefficient_of_variation std/sigma
            RS_temp = 10. * torch.zeros(all_sampled_rgb.shape[1]).cuda()
            tot_sampled_rgb = torch.concat([all_sampled_rgb,anchored_rgb],dim=0)
            tot_d_mask = torch.concat([all_d_mask,torch.ones(1,all_d_mask.shape[-1]).bool().cuda()])
            mean_color = tot_sampled_rgb.sum(0,keepdim=True)/tot_d_mask.sum(0,keepdim=True)[:,:,None,None]
            std_color = (((tot_sampled_rgb-mean_color)**2)*(tot_d_mask.float()[...,None,None])).sum(0)/(tot_d_mask.sum(0)[:,None,None])**0.5 # (-1,P^2,3)
            coef_var = std_color/(mean_color.squeeze(0)+1e-8)
            val_inds_fina = (mean_color>1e-5).squeeze().all(-1).all(-1)
            RS_temp[val_inds_fina] = coef_var.sum(-1).mean(-1)[val_inds_fina]

            save_dir = f"debug_refreal/{iteration}"
            os.makedirs(save_dir,exist_ok=True)
            RS_temp_vis = RS_temp.reshape((H,W))/RS_temp.max()
            save_image(RS_temp_vis,f"{save_dir}/{ref_image_name}_ref_score_cv.png")
        except:
            pass
        ## post process

        # import ipdb;ipdb.set_trace()
        ## 计算score
        RS_temp = 10. * torch.zeros(all_sampled_rgb.shape[1]).cuda()
        diff_color = torch.zeros_like(all_sampled_rgb).cuda() # (N,-1,P^2,3)
        diff_color[all_d_mask] = torch.abs(all_sampled_rgb-anchored_rgb.expand(all_sampled_rgb.size()))[all_d_mask]
        val_mean = torch.sum(diff_color, dim=0) / (all_d_mask.sum(0) + 1e-8)[:,None,None]
        val_inds_fina = (all_d_mask.sum(0)>0)
        RS_temp[val_inds_fina] = val_mean.sum(-1).mean(-1)[val_inds_fina]


        try:
            RS_temp_remove_edges = torch.zeros_like(RS_temp)
            edges_val_inds = edges_mask.reshape(*RS_temp.shape).bool()
            RS_temp_remove_edges[edges_val_inds] = RS_temp[edges_val_inds]
        except:
            pass

        def vis_ints_conts(all_d_mask,all_sampled_rgb,H,W):
            save_dir = f"debug_refreal/{iteration}"
            os.makedirs(save_dir,exist_ok=True)
            tmask_vis = all_d_mask.reshape(-1,H,W).unsqueeze(1).repeat((1,3,1,1))
            tmask_grid = make_grid(tmask_vis.float(),nrow=4)
            save_image(tmask_grid,f"{save_dir}/{ref_image_name}_tmask_vis.png")
            trgb_vis = all_sampled_rgb[:,:,4,:].reshape(-1,H,W,3).permute((0,3,1,2))
            tsrgb_grid = make_grid(trgb_vis,nrow=4)
            save_image(tsrgb_grid,f"{save_dir}/{ref_image_name}_tsrgb_grid.png")
            RS_temp_vis = RS_temp.reshape((H,W))/RS_temp.max()
            save_image(RS_temp_vis,f"{save_dir}/{ref_image_name}_ref_score.png")
            save_image(edges_mask.reshape(H,W),f"./debug_ref/{iteration}_edges.png")
            save_image(anchored_rgb[:,:,4,:].reshape(-1,H,W,3).permute((0,3,1,2)),f"{save_dir}/{ref_image_name}_anchored_rgb.png")


        try:
            vis_ints_conts(all_d_mask,all_sampled_rgb,H,W)
        except:
            print("Visualization Erorr In Reflection Score Calc")

        if edges_aware:
            ref_score_images[viewpoint_cam.image_name] = RS_temp_remove_edges
        else:
            ref_score_images[viewpoint_cam.image_name] = RS_temp

    return ref_score_images
        # def mahalanobis_distance():
        #     pass

        # def reflection_score(ref_rgb_vals, nst_rgb_vals:torch.Tensor, d_mask_vals:torch.Tensor, is_edge:bool):
        #     if is_edge: return 0
        #     if d_mask_vals.sum() == 0: return 0
        #     nst_rgb_vals = nst_rgb_vals[d_mask_vals]
        #     rgb_vals = torch.cat([ref_rgb_vals,nst_rgb_vals],dim=0).permute((1,0,2))
        #     rgb_std = rgb_vals.std(dim=1).sum(-1).mean(0)

        #     return rgb_std
        
        # with Parallel(n_jobs=64) as parallel:
        #     results = parallel()(
        #         delayed(reflection_score)
        #         (
        #             ref_rgb[:,idx,:], all_sampled_rgb[:,:,idx,:], all_d_mask[:,idx], (edges_mask[idx] > 0.99) 
        #         )
        #         for idx in range(H*W)
        #     )
        #     ref_score_images[viewpoint_cam.image_name] = torch.tensor(results).reshape(H,W)

    #### 后处理

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, model_path, args, debug_from=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()
    # Set up parameters 
    TOT_ITER = opt.iterations + 1
    TEST_INTERVAL = 1000
    MESH_EXTRACT_INTERVAL = 2000
    ENV_GAUSSIAN_START_ITER = 20000

    # For real scenes
    USE_ENV_SCOPE = opt.use_env_scope  # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4


    gaussians = GaussianModel(dataset.sh_degree)
    env_gaussians = EnvGaussianModel(dataset.sh_degree)
    indirect_renderer = HardwareRendering()
    set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter)) # #
    scene = Scene(dataset, gaussians, env_gaussians=env_gaussians)  # init all parameters(pos, scale, rot...) from pcds
    gaussians.training_setup(opt)
    if checkpoint:
        iters = int(checkpoint.split("/")[-1].replace(".pth","").replace("chkpnt",""))
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.load_envlight(os.path.dirname(checkpoint),iters,dataset)
        gaussians.restore(model_params, opt)
        
        if iters >= opt.indirect_from_iter:
            mesh_iters = (iters // 2000) * 2000
            mesh_path = os.path.join(os.path.join(os.path.dirname(checkpoint),f"test_{mesh_iters:06d}.ply"))
            mesh = o3d.io.read_triangle_mesh("/data14_2/tjm/code/ref-gaussian/output_envgs/gardenspheres/gardenspheres-0420_0529/test_010000.ply")
            gaussians.update_mesh(mesh)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussExtractor = GaussianExtractor(gaussians, render_surfel, pipe, bg_color=bg_color) 

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_normal_smooth_for_log = 0.0
    ema_depth_smooth_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_loss_geo_loss = 0.0
    ema_loss_pho_loss = 0.0
    ema_loss_base_color = 0.0
    ema_loss_metallic_warp = 0.0
    ema_loss_roughness_warp = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print(f'Propagation until: {opt.normal_prop_until_iter }')
    print(f'Densify until: {opt.densify_until_iter}')
    print(f'Total iterations: {TOT_ITER}')

    opt.mesh_res = 1024
    initial_stage = opt.initial
    if not initial_stage:
        opt.init_until_iter = 0

    # load prior
    scene_name = dataset.source_path.split("/")[-1]
    mtl_images, rgh_images, albedo_images = None, None, None
    mask_images, normal_images, mtl_images, rgh_images, albedo_images, ref_score_images = load_normal_prior(dataset, scene_name, dataset.resolution, args=args)
    normal_gamma = 2

    # ref_score_images = None
    has_init_indirect = False
    env_flag = False

    # Training loop
    while iteration < TOT_ITER:
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if (first_iter > ENV_GAUSSIAN_START_ITER and iteration==first_iter) or iteration == ENV_GAUSSIAN_START_ITER:
            print(f"Env Gaussian Initialization...")
            anchored_lst = []
            env_gaussians.restore_from_refgs(gaussians.capture(),opt,anchored_lst=anchored_lst)
        #     env_gaussians.freeze_geo()
        #     env_flag = True
        # if iteration >= 15000:
        #     env_gaussians.unfreeze_geo()
        #     env_flag = True

        # Increase SH levels every 1000 iterations
        if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
            # if iteration <= opt.feature_rest_from_iter + 3000:
            gaussians.oneupSHdegree()

        # Control the init stage
        if iteration > opt.init_until_iter:
            initial_stage = False
        
        # Control the indirect stage
        if iteration == opt.indirect_from_iter + 1:
            opt.indirect = 1
            env_flag = True
            # reset_gaussian_para2(gaussians, opt)
            # reset_gaussian_para3(gaussians, opt)
            # gaussians.frozen_gaussian_gemotry()
            # gaussians.rstSHdegree()
        
        # if (iteration > opt.indirect_from_iter + 10000) and (iteration - opt.indirect_from_iter) % 1000 == 0:
        #     gaussians.oneupSHdegree()
            
        
        # if iteration == opt.indirect_from_iter + 500:
        #     gaussians.restore_gaussian_gemotry_lr()

        # Init Some Guassian Parameters
        # if (not has_init_indirect) or iteration == opt.indirect_from_iter + 1:
        #     gaussians.init_indirect_learning_stage()

        # normal gamma
        if iteration > opt.init_until_iter:
            normal_gamma = 1
        if iteration > 7000:
            normal_gamma = 0
        if iteration > opt.normal_prop_until_iter:
            normal_gamma = 0
        if iteration > opt.densify_until_iter:
            normal_gamma = 0
        if iteration > opt.indirect_from_iter and iteration < opt.indirect_from_iter + 10000:
            normal_gamma = 0


        if iteration == (opt.volume_render_until_iter + 1) and opt.volume_render_until_iter > opt.init_until_iter:
            reset_gaussian_para(gaussians, opt)


        # Initialize envmap
        if not initial_stage:
            if iteration <= opt.volume_render_until_iter:
                envmap2 = gaussians.get_envmap_2 
                envmap2.build_mips()
            else:
                envmap = gaussians.get_envmap 
                envmap.build_mips()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


        # Set render
        render = select_render_method(iteration, opt, initial_stage, indirect_renderer, env_gaussians)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        
        image_weight = (1.0 - get_img_grad_weight(gt_image))
        image_weight = (image_weight).clamp(0,1).detach() ** 2
        
        def get_current_normal_loss_weight(current,opt):
            return 0.05
            if current < 10000:
                return 0.015
            if current < 15000:
                return 0.05
            if current < 20000:
                return 0.1
            if current < 25000:
                return 0.15
            return 0.1
            # if current < 15_000:
            #     if current < opt.normal_loss_start:
            #         return 0.015
            #     start,end = opt.normal_loss_start, 15_000
            #     return 0.015 +  (min(current,end) - start) / (end - start) * (0.08 - 0.015)
            # else:
            #     start,end = 15_000, 25_000
            #     return 0.08 - (min(current,end) - start) / (end - start) * (0.08 - 0.015)
            
        opt.lambda_normal_render_depth = get_current_normal_loss_weight(iteration,opt)
        total_loss, tb_dict = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration, image_weight, mask_images[viewpoint_cam.image_name])
        dist_loss, normal_loss, loss, Ll1, normal_smooth_loss, depth_smooth_loss = tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss0"], tb_dict["loss_l1"], tb_dict["loss_normal_smooth"], tb_dict["loss_depth_smooth"] 

        # normal_prior
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        ref_mask = None

        if not initial_stage and iteration > opt.volume_render_until_iter:
            if mask_images[viewpoint_cam.image_name] is not None:
                image_mask = mask_images[viewpoint_cam.image_name].cuda()
            rendered_opacity = render_pkg["rend_alpha"]
            o = rendered_opacity.clamp(1e-6, 1 - 1e-6).view_as(image_mask)
            loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
            tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
            total_loss = total_loss + 0.01 * loss_mask_entropy

            # total_loss += 0.1 * torch.abs(1-render_pkg["rend_alpha"]).mean()
            refl_strength_map = render_pkg["refl_strength_map"]
            roughness_map = render_pkg["roughness_map"]
            albedo_map = render_pkg["base_color_map"]
            # ref_mask = refl_strength_map.reshape(-1,1).detach().clone()
            # ref_mask[ref_mask<0.35] = 0.
            
            # if iteration > 12000 and (iteration % 2000) ==0:
            #     import ipdb;ipdb.set_trace()

            # roughness_loss,ref_loss,albedo_loss = cal_material_loss(
            #     viewpoint_cam, roughness_map, refl_strength_map, albedo_map, mask_images, mtl_images, rgh_images, albedo_images
            # )
            # roughness_gamma = 2#1
            # ref_gamma = 2#1
            # albedo_gamma = 4#1
            # total_loss += 0.001 * roughness_gamma * roughness_loss
            # total_loss += 0.001 * ref_gamma * ref_loss
            # total_loss += 0.001 * albedo_gamma * albedo_loss
        HSV_mask = None
        # if iteration > 0:
        #     HSV_mask = get_HSV_mask(gt_image,mask_images,viewpoint_cam)
        # import ipdb;ipdb.set_trace()
        l1_normal, cos_normal, l1_normal2, cos_normal2 = mono_normal_loss(
            viewpoint_cam, surf_normal, rend_normal, mask_images, normal_images, normal_gamma, iteration, ref_mask, HSV_mask
        )
        tb_dict.update({
            "mono_rend_normal": 0.01 * (l1_normal + cos_normal),
            "mono_surf_normal": 0.01 * (l1_normal2 + cos_normal2)
        })
        total_loss += normal_gamma *  0.01 * (l1_normal + cos_normal + l1_normal2 + cos_normal2)
        # total_loss += normal_gamma *  0.01 * (l1_normal + cos_normal)
        # total_loss += normal_gamma *  0.01 * (l1_normal2 + cos_normal2)

        d_mask = ref_weight = None
        geo_loss = ncc_loss = None
        ncc_loss_basecolor = None
        base_color_loss_map = None
        metallic_warp_loss = None
        roughness_warp_loss = None
        if iteration > 25000:
            geo_loss, ncc_loss, ncc_loss_basecolor, metallic_warp_loss, roughness_warp_loss ,d_mask, ref_weight, base_color_loss_map = calc_warp_loss(viewpoint_cam=viewpoint_cam,
                scene=scene,
                opt=opt,gaussians=gaussians,dataset=dataset,pipe=pipe,
                render=render,
                render_pkg=render_pkg,
                albeldo_images=albedo_images,
                mtl_images=mtl_images,
                rgh_images=rgh_images,
                mask_images=mask_images,
                iteration=iteration,debug_path=None,bg=background,
                use_metallic_warp=opt.use_metallic_warp_loss,
                use_roughness_warp=opt.use_roughness_warp_loss
            )
            if geo_loss is not None:
                total_loss += geo_loss
            if ncc_loss is not None:
                total_loss += ncc_loss
            
            if iteration > 10000 and ncc_loss_basecolor is not None:
                total_loss += ncc_loss_basecolor
            if iteration > opt.rghmtl_warp_loss_start_iter and metallic_warp_loss is not None:
                total_loss += metallic_warp_loss
            if iteration > opt.rghmtl_warp_loss_start_iter and roughness_warp_loss is not None:
                total_loss += roughness_warp_loss
        
        # #### supervise material learning with ref_score
        # if not initial_stage and iteration > opt.ref_score_start_iter:
        #     ref_score_image = ref_score_images[viewpoint_cam.image_name]
        #     refl_strenghth_map = render_pkg["refl_strength_map"]
        #     roughness_map = render_pkg["roughness_map"]
        #     ref_score_loss_metallic = torch.abs(refl_strenghth_map - 1.0)[ref_score_image]
        #     ref_score_loss_roughness = torch.abs(roughness_map)[ref_score_image]
        #     if (ref_score_loss_metallic>opt.tel_thres).sum() > 0:
        #         total_loss += opt.ref_score_loss_weight * ref_score_loss_metallic[ref_score_loss_metallic>opt.tel_thres].mean()
        #     if (ref_score_loss_roughness>opt.tel_thres).sum() > 0:
        #         total_loss += opt.ref_score_loss_weight * ref_score_loss_roughness[ref_score_loss_roughness>opt.tel_thres].mean()
            
        #     # minus_ref_score_image = torch.bitwise_and(~mask_images[viewpoint_cam.image_name].view_as(ref_score_image),~ref_score_image)
        #     minus_ref_score_image = ~ref_score_image
        #     ref_score_loss_metallic2 = torch.abs(refl_strenghth_map)[minus_ref_score_image]
        #     ref_score_loss_roughness2 = torch.abs(1. - roughness_map)[minus_ref_score_image]
        #     ref_score_loss_metallic2 = ref_score_loss_metallic2.mean()
        #     ref_score_loss_roughness2 = ref_score_loss_roughness2.mean()
        #     if (ref_score_loss_metallic2>opt.tel_thres).sum() > 0:
        #         total_loss += opt.ref_score_loss_weight * ref_score_loss_metallic2[ref_score_loss_metallic2>opt.tel_thres].mean()
        #     if (ref_score_loss_roughness2>opt.tel_thres).sum() > 0:
        #         total_loss += opt.ref_score_loss_weight * ref_score_loss_roughness2[ref_score_loss_roughness2>opt.tel_thres].mean()            
        # #### End ref_score loss

        ### Env Supervision
        # if iteration % 10 == 0 and iteration > 13000:
        #     env_render_pkg = render_initial(viewpoint_cam, env_gaussians, pipe, background, srgb=opt.srgb, opt=opt)
        #     env_gs_depth = env_render_pkg["surf_depth"]
        #     env_gs_normal = env_render_pkg["rend_normal"]
        #     env_gs_rend = env_render_pkg["render"]
        #     env_depth_loss = None
        #     env_normal_loss = None
        #     if iteration > 13000:
        #         env_depth_loss = torch.abs(env_gs_depth - render_pkg['surf_depth'].detach())
        #         env_normal_loss = torch.abs(env_gs_normal - render_pkg['rend_normal'].detach())
        #     total_loss += 0.01 * (env_depth_loss.mean() + env_normal_loss.mean())
        ### End Env Supervision

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2
        
        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            total_loss += REFL_MSK_LOSS_W * refl_msk_loss
        
        # with torch.autograd.detect_anomaly():        
        total_loss.backward()

        iter_end.record()


        with torch.no_grad():

            VISUAL_INTERVAL = 1000
            if iteration % VISUAL_INTERVAL == 0 or iteration == first_iter + 1 or iteration == opt.volume_render_until_iter + 1:
                save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, iteration, initial_stage, d_mask, ref_weight, base_color_loss_map,render_pkg, None)
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            ema_normal_smooth_for_log = 0.4 * normal_smooth_loss + 0.6 * ema_normal_smooth_for_log
            ema_depth_smooth_for_log = 0.4 * depth_smooth_loss + 0.6 * ema_depth_smooth_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            if geo_loss is not None: ema_loss_geo_loss = 0.4 * geo_loss + 0.6 * ema_loss_geo_loss
            if ncc_loss is not None: ema_loss_pho_loss = 0.4 * ncc_loss + 0.6 * ema_loss_pho_loss
            if ncc_loss_basecolor is not None: ema_loss_base_color = 0.4 * ncc_loss_basecolor + 0.6 * ema_loss_base_color
            if metallic_warp_loss is not None: ema_loss_metallic_warp = 0.4 * metallic_warp_loss + 0.6 * ema_loss_metallic_warp
            if roughness_warp_loss is not None: ema_loss_roughness_warp = 0.4 * roughness_warp_loss + 0.6 * ema_loss_roughness_warp

            if iteration % TEST_INTERVAL == 0:
                psnr_test = evaluate_psnr(scene, render, {"pipe": pipe, "bg_color": background, "opt": opt})
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Distort": f"{ema_dist_for_log:.{5}f}",
                    # "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Geo": f"{ema_loss_geo_loss:.{5}f}",
                    "Pho": f"{ema_loss_pho_loss:.{5}f}",
                    "BC": f"{ema_loss_base_color:.{5}f}",
                    "MW": f"{ema_loss_metallic_warp:.{5}f}",
                    "RW": f"{ema_loss_roughness_warp:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "Ref": f"{gaussians.get_refl_strength_to_total * 100:.{4}f}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == TOT_ITER:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/geo_loss', ema_loss_geo_loss, iteration)
                tb_writer.add_scalar('train_loss_patches/pho_loss', ema_loss_pho_loss, iteration)
                tb_writer.add_scalar('train_loss_patches/bc_loss',ema_loss_base_color, iteration)


            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, {"pipe": pipe, "bg_color": background, "opt":opt},dataset=dataset)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification for EnvGS
            if "indirect_out" in render_pkg.keys() and env_flag:
                env_gaussians.update_env_gs(iteration,opt,scene,render_pkg["indirect_out"])
                # if iteration % 400 == 0:
                #     # import ipdb;ipdb.set_trace()
                #     render_pkg = render_initial(viewpoint_cam, env_gaussians, pipe, background, srgb=opt.srgb, opt=opt)
                #     env_gs_depth = render_pkg["surf_depth"]
                #     env_gs_normal = render_pkg["rend_normal"]
                #     env_gs_rend = render_pkg["render"]
                #     save_image(visualize_depth(env_gs_depth),f"env_debug/{iteration:06d}_env_gs_depth.png")
                #     save_image(env_gs_normal*0.5+0.5,f"env_debug/{iteration:06d}_env_gs_normal.png")
                #     save_image(env_gs_rend,f"env_debug/{iteration:06d}_env_gs_rend.png")


            # Densification
            if iteration < opt.densify_until_iter and iteration != opt.volume_render_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration <= opt.init_until_iter:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= opt.normal_prop_until_iter :
                    opacity_reset_intval = 3000
                    densification_interval = opt.densification_interval_when_prop
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent,
                                                size_threshold)

                HAS_RESET0 = False
                if iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity0()
                    if iteration > opt.indirect_from_iter:
                        gaussians.reset_refl(exclusive_msk=outside_msk,rst_value=0.1)
                    else:
                        gaussians.reset_refl(exclusive_msk=outside_msk)
                if opt.opac_lr0_interval > 0 and (
                        opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.opac_lr0_interval == 0:
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if (opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.normal_prop_interval == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        if iteration > opt.volume_render_until_iter and opt.volume_render_until_iter > opt.init_until_iter:
                            gaussians.dist_color(exclusive_msk=outside_msk)
                            # gaussians.dist_albedo(exclusive_msk=outside_msk)

                        gaussians.reset_scale(exclusive_msk=outside_msk)
                        if opt.opac_lr0_interval > 0 and iteration != opt.normal_prop_until_iter :
                            gaussians.set_opacity_lr(0.0)
                
            if (iteration >= opt.indirect_from_iter and iteration % MESH_EXTRACT_INTERVAL == 0) or iteration == (opt.indirect_from_iter):
                if not HAS_RESET0:
                    gaussExtractor.reconstruction(scene.getTrainCameras(),opt=opt)
                    if 'refreal' in dataset.source_path or 'tnt' in dataset.source_path:
                        mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    else:
                        depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
                        voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
                        sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
                        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
                    ply_path = os.path.join(model_path,f'test_{iteration:06d}.ply')
                    o3d.io.write_triangle_mesh(ply_path, mesh)
                    gaussians.update_mesh(mesh)

            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                if env_flag:
                    env_gaussians.optimizer.step()
                    env_gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

        iteration += 1








# ============================================================
# Utils for training

from functools import partial
def select_render_method(iteration, opt, initial_stage, indirect_renderer, env):

    if initial_stage:
        render = render_initial
    elif iteration <= opt.volume_render_until_iter:
        render = render_volume
    elif iteration <= opt.indirect_from_iter:
        render = render_surfel
    else:
        render =  partial(render_surfel2,indirect_renderer,env)

    return render


def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_ori_color()
    # gaussians.reset_refl_strength(opt.init_refl_value)
    gaussians.reset_refl_strength(0.1)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr

def reset_gaussian_para2(gaussians, opt):
    gaussians.reset_ori_color()
    gaussians.reset_refl_strength(0.1)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.reset_features()

    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr


def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, initial_stage, d_mask, ref_weight, base_color_loss_map,render_pkg=None,ref_score=None):
    with torch.no_grad():
        if render_pkg is None:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, srgb=opt.srgb, opt=opt)

        error_map = torch.abs(viewpoint_cam.original_image.cuda() - render_pkg["render"])
        rot = viewpoint_cam.R.T.float().cuda()
        
        def a(x):
            sz=x.shape
            x = (rot@x.reshape(3,-1)).reshape(*sz)[[2,1,0],...]
            return 0.5 * x + 0.5
        if initial_stage:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),
                render_pkg["render"], 
                render_pkg["rend_alpha"].repeat(3, 1, 1),
                visualize_depth(render_pkg["surf_depth"]),  
                a(render_pkg["rend_normal"]), 
                a(render_pkg["surf_normal"]), 
                error_map 
            ]

        elif iteration <= opt.volume_render_until_iter:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"], 
                render_pkg["base_color_map"], 
                render_pkg["diffuse_map"],      
                render_pkg["specular_map"],  
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]), 
                a(render_pkg["rend_normal"]),  
                a(render_pkg["surf_normal"]), 
                error_map
            ]
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["direct_light"],
                    render_pkg["indirect_light"],
                ]

        else:
            visualization_list = [
                viewpoint_cam.original_image.cuda(),  
                render_pkg["render"],  
                render_pkg["base_color_map"],  
                render_pkg["diffuse_map"],
                render_pkg["specular_map"],
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]),  
                a(render_pkg["rend_normal"]),  
                a(render_pkg["surf_normal"]),  
                error_map, 
            ]
        
        if "rend_distance" in render_pkg.keys():
            visualization_list.append(
                visualize_depth(render_pkg["rend_distance"]),
            )
        if d_mask is not None:
            visualization_list.append(
                visualize_d_mask(d_mask,render_pkg["rend_normal"].shape[1],render_pkg["rend_normal"].shape[2])
            )
        if ref_weight is not None:
            visualization_list.append(visualize_depth(ref_weight,norm=False))

        if base_color_loss_map is not None:
            visualization_list.append(visualize_depth2(base_color_loss_map))

        if "visibility" in render_pkg.keys():
            visualization_list += [
                render_pkg["visibility"].repeat(3, 1, 1),
                render_pkg["direct_light"],
                render_pkg["indirect_light"],
            ]
        if "indirect_out" in render_pkg.keys():
            visualization_list += [
                render_pkg["indirect_out"]["render"],
                render_pkg["indirect_out"]["specular"].repeat(3, 1, 1),
                render_pkg["indirect_out"]["rend_alpha"].repeat(3,1,1)
            ]
        if ref_score is not None:
            visualization_list.append(ref_score.repeat(3,1,1))

        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 2400
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        if not initial_stage:
            if opt.volume_render_until_iter > opt.init_until_iter and iteration <= opt.volume_render_until_iter:
                env_dict = gaussians.render_env_map_2() 
            else:
                env_dict = gaussians.render_env_map()

            grid = [
                env_dict["env1"].permute(2, 0, 1),
                env_dict["env2"].permute(2, 0, 1),
            ]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    
    # backup import files
    os.system(f"cp -rf ./gaussian_renderer {args.model_path}/gaussian_renderer")
    os.system(f"cp -rf ./scene {args.model_path}/scene")
    os.system(f"cp -rf ./utils {args.model_path}/utils")
    os.system(f"cp -rf ./arguments {args.model_path}/arguments")
    sh_files = glob("*.sh")
    for sh_file in sh_files:
        os.system(f"cp {sh_file} {args.model_path}/{sh_file}")
    train_files = glob("train*.py")
    for train_file in train_files:
        os.system(f"cp -rf {train_file} {args.model_path}/{train_file}")

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderkwargs,dataset=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if config['name']=="test":
                    save_psnr(iteration,dataset.model_path, (psnr_test).item())
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs):
    psnr_test = 0.0
    torch.cuda.empty_cache()
    os.makedirs("./output_data10_2/evaluate", exist_ok=True)

    if len(scene.getTestCameras()):
        for viewpoint in scene.getTestCameras():
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            try:
                save_image(image, "./output_data10_2/evaluate/"+viewpoint.image_name+".png")
            except:
                pass
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(scene.getTestCameras())
        
    torch.cuda.empty_cache()
    return psnr_test

@torch.no_grad()
def save_psnr(iteration,model_path, psnr_test):
    import json
    psnr_file = os.path.join(model_path, "psnr.json")
    if os.path.exists(psnr_file):
        with open(psnr_file, 'r') as f:
            psnr_dict = json.load(f)
    else:
        psnr_dict = {}
    psnr_dict[str(iteration)] = psnr_test
    with open(psnr_file, 'w') as f:
        f.write(json.dumps(psnr_dict))



# ============================================================================
# Main function

seed = 3407

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    np.random.seed(seed) # Numpy module.
    random.seed(seed) # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(seed)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,15000,20000,30000,40000,50000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000,12000,15000,20000,30000,40000,50000,60000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Prior paths
    parser.add_argument("--geowizard_path", type=str, default=None)
    parser.add_argument("--metric3d_path", type=str, default=None)
    parser.add_argument("--idarb_path", type=str, default=None)
    parser.add_argument("--ref_score_path", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 2500)]
    args.test_iterations.append(args.volume_render_until_iter)

    
    if not args.model_path:
        current_time = datetime.now().strftime('%m%d_%H%M')
        last_subdir = os.path.basename(os.path.normpath(args.source_path))
        
        args.model_path = os.path.join(
            "./output_glossy", f"{last_subdir}",
            f"{last_subdir}-{current_time}"
        )

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.model_path, args)

    # All done
    print("\nTraining complete.")