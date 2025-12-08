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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCorrect,fov2focal

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 data_device = "cuda", HWK = None, gt_refl_mask = None, gray_image = None, ncc_scale=1.0
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.nearest_id = []
        self.nearest_names = []
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.refl_mask = gt_refl_mask
        self.ncc_scale = ncc_scale
        # import ipdb;ipdb.set_trace()
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if HWK is not None: # #
            self.image_width = int(HWK[1])
            self.image_height = int(HWK[0])
        
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0)#.to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                # self.original_image *= gt_alpha_mask.to(self.data_device)
                self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width))#, device=self.data_device)
                self.gt_alpha_mask = None
            
            if gray_image is not None:
                self.original_image_gray = gray_image
            else:
                self.original_image_gray = (0.299 * self.original_image[0] + 0.587 * self.original_image[1] + 0.114 * self.original_image[2])[None]

        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        # NOTE: if camera has significant center offset (cx,cy!=W/2,H/2), MUST USE getProjectionMatrixCorrect !!
        #HWK = None
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if HWK is None:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        else:
            self.HWK = HWK
            self.projection_matrix = getProjectionMatrixCorrect(self.znear, self.zfar, HWK[0], HWK[1], HWK[2]).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.R = torch.tensor(self.R, dtype=torch.float32, device='cuda')
        self.T = torch.tensor(self.T, dtype=torch.float32, device='cuda')

    def get_image(self):
        return self.original_image.cuda(), self.original_image_gray.cuda()

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    
    def get_rays(self, scale=1.0): # 获取每个像素的射线
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

