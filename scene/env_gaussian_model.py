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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.refl_utils import safe_normalize,flip_align_view

class EnvGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_weight_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.max_gs = 2e6
        self.max_gs_threshold = 0.9
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_weight_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        xyz_weight_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_weight_accum = xyz_weight_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_xyz_weight_avg(self):
        avg = self.xyz_weight_accum / self.denom
        avg[avg.isnan()] = 0.0
        return avg
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args, lr_downfactor_geo=5., anchored_lst=[]):
        self.percent_dense = 0.01#training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_weight_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.features_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.features_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # if "all" in anchored_lst:
        #     anchored_lst = ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]
        # new_l = []
        # for param_group in l:
        #     if param_group["name"] not in anchored_lst:
        #         new_l.append(param_group)
        # l = new_l

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_stats(self,mask):
        valid_points_mask = ~mask
        self.xyz_gradient_accum.set_(self.xyz_gradient_accum[valid_points_mask])
        self.denom.set_(self.denom[valid_points_mask])
        self.max_radii2D.set_(self.max_radii2D[valid_points_mask])
        self.xyz_weight_accum.set_(self.xyz_weight_accum[valid_points_mask])
        assert self.xyz_gradient_accum.shape[0] == self.get_xyz.shape[0]
        assert self.denom.shape[0] == self.get_xyz.shape[0]
        assert self.max_radii2D.shape[0] == self.get_xyz.shape[0]
        assert self.xyz_weight_accum.shape[0] == self.get_xyz.shape[0]

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_weight_accum = self.xyz_weight_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.xyz_weight_accum = torch.zeros((self.get_xyz.shape[0],1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def densify_stats(self, selected_pts_mask: torch.Tensor, split: int = 1, ratio: float = 1.0):
        new_xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, self.xyz_gradient_accum[selected_pts_mask].repeat(split, 1) * ratio], dim=0)
        new_denom = torch.cat([self.denom, self.denom[selected_pts_mask].repeat(split, 1)], dim=0)
        new_max_radii2D = torch.cat([self.max_radii2D, self.max_radii2D[selected_pts_mask].repeat(split) * ratio], dim=0)
        new_xyz_weight_accum = torch.cat([self.xyz_weight_accum, self.xyz_weight_accum[selected_pts_mask].repeat(split, 1) * self.xyz_weight_accum.max()], dim=0)
        self.xyz_gradient_accum.set_(new_xyz_gradient_accum)
        self.denom.set_(new_denom)
        self.max_radii2D.set_(new_max_radii2D)
        self.xyz_weight_accum.set_(new_xyz_weight_accum)
        assert self.xyz_gradient_accum.shape[0] == self.get_xyz.shape[0]
        assert self.denom.shape[0] == self.get_xyz.shape[0]
        assert self.max_radii2D.shape[0] == self.get_xyz.shape[0]
        assert self.xyz_weight_accum.shape[0] == self.get_xyz.shape[0]

    def densify_and_split(self, grads, grad_threshold,scene_extent,split_screen_threshold=None,N=2):
        # n_init_points = self.get_xyz.shape[0]
        # # Extract points that satisfy the gradient condition
        # padded_grad = torch.zeros((n_init_points), device="cuda")
        # padded_grad[:grads.shape[0]] = grads.squeeze()
        grads = self.get_xyz_gradient_avg()
        high_grads = (grads >= grad_threshold).squeeze(-1)
        # high_grads = torch.where(grads >= grad_threshold, True, False)
        selected_pts_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        if split_screen_threshold is not None:
            selected_pts_mask = torch.logical_or(selected_pts_mask, self.max_radii2D > split_screen_threshold)
        selected_pts_mask = torch.logical_and(selected_pts_mask, high_grads)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            #   torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        n_split = selected_pts_mask.sum().item()
        if n_split>0:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
            self.densify_stats(selected_pts_mask, N, 1.0 / (0.8 * N))
            
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * n_split, device="cuda", dtype=bool)))
            self.prune_points(prune_filter)
        torch.cuda.empty_cache()

    def split(self,mask,N=2,ratio=0.8):
        stds = self.get_scaling[mask].repeat(N, 1)  # (M * N, 2), NOTE: 2DGS has only 2 scaling parameters
        stds = torch.cat([stds, torch.zeros_like(stds[:, :1])], dim=-1)  # (M * N, 3)
        means = torch.zeros_like(stds)  # (M * N, 3)
        # Only split along the longest axis to avoid floaters and make the optimization more stable
        samples = torch.normal(means, stds).to(device="cuda", dtype=self.get_xyz.dtype)  # (M * N, 3)
        rots = build_rotation(self._rotation[mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[mask].repeat(N, 1) / (ratio * N))  # NOTE: 2DGS has only 2 scaling parameters
        # Split features
        new_rotation = self._rotation[mask].repeat(N, 1)
        new_features_dc = self._features_dc[mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[mask].repeat(N, 1, 1)
        new_opacity = self._opacity[mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self.densify_stats(mask, N, 1.0 / (ratio * N))
        # Prune splited points
        n_split = mask.sum().item()
        prune_mask = torch.cat((mask, torch.zeros((n_split * N,), device="cuda", dtype=bool)))
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        self.densify_stats(selected_pts_mask,1,1.)

    def reset_stats(self):
        device = self.get_xyz.device
        self.xyz_gradient_accum.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.denom.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.max_radii2D.set_(torch.zeros((self.get_xyz.shape[0]), device=device))
        self.xyz_weight_accum.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))

    def get_xyz_gradient_avg(self):
        avg = self.xyz_gradient_accum / self.denom
        avg[avg.isnan()] = 0.0
        return avg

    def prune_min_opacity_and_gradients(
        self,
        min_opacity: float = None,
        min_gradient: float = None,
        prefix: str = ''
    ):
        n_before = self.get_xyz.shape[0]

        if min_opacity is not None:
            min_occs = (self.get_opacity < min_opacity).squeeze(-1)
            n_min_occ = min_occs.sum().item()
        else:
            min_occs = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_min_occ = 0
        if min_gradient is not None:
            grads = self.get_xyz_gradient_avg()
            min_grads = ((grads <= min_gradient) & (self.denom != 0)).squeeze(-1)
            n_min_grad = min_grads.sum().item()
        else:
            min_grads = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_min_grad = 0

        prune_mask = torch.logical_or(min_occs, min_grads)
        if prune_mask.sum().item() > 0:
            self.prune_points(prune_mask)
            torch.cuda.empty_cache()

        n_after = self.get_xyz.shape[0]

    def prune_max_scene_and_screen(
        self,
        max_scene_threshold: float = None,
        max_screen_threshold: float = None,
        min_weight_threshold: float = None,
        extent=None,
    ):
        n_before = self.get_xyz.shape[0]

        if max_screen_threshold is not None:
            max_screens = self.max_radii2D > max_screen_threshold
            n_max_screen = max_screens.sum().item()
        else:
            max_screens = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_max_screen = 0
        if max_scene_threshold is not None:
            max_scenes = torch.max(self.get_scaling, dim=-1).values > extent * max_scene_threshold
            n_max_scene = max_scenes.sum().item()
        else:
            max_scenes = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_max_scene = 0
        if min_weight_threshold is not None:
            # Accumulated weight related prune/split mask
            weights = self.get_xyz_weight_avg()
            min_weights = (weights < torch.quantile(weights, min_weight_threshold)).squeeze(-1)
            n_min_weight = min_weights.sum().item()
        else:
            min_weights = torch.ones((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            n_min_weight = 0

        # Get the prune and split mask respectively
        prune_mask = torch.logical_or(max_screens, max_scenes)
        split_mask = torch.logical_and(prune_mask, ~min_weights)
        prune_mask = torch.logical_and(prune_mask, min_weights)
        split_mask = split_mask[~prune_mask]
        n_prune = prune_mask.sum().item()
        n_split = split_mask.sum().item()

        # Actual pruning
        if n_prune > 0:
            self.prune_points(prune_mask)
            torch.cuda.empty_cache()
        # Actual splitting
        if n_split > 0:

            self.split(split_mask, 5, 0.5)

    def prune_visibility(self):
        n_before = self.get_xyz.shape[0]
        n_after = int(self.max_gs * self.max_gs_threshold)
        n_prune = n_before - n_after

        if n_prune > 0:
            weights = self.get_xyz_weight_avg()
            # Find the mask of top n_prune smallest `self.xyz_weight_accum`
            _, indices = torch.topk(weights[..., 0], n_prune, largest=False)
            prune_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            prune_mask[indices] = True
            # Prune points
            self.prune_points(prune_mask)
            torch.cuda.empty_cache()


    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size=None,
        split_screen_threshold=None,
    ):
        grads = self.get_xyz_gradient_avg()
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, split_screen_threshold)
        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        # if max_screen_size:
        #     big_points_vs = self.max_radii2D > max_screen_size
        #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # self.prune_points(prune_mask)
        self.prune_min_opacity_and_gradients(min_opacity, None)
        self.prune_max_scene_and_screen(0.1, max_screen_size, 0.1, extent)
        self.prune_visibility()
        self.reset_stats()
        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter, weight_accumulate=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if weight_accumulate is not None: self.xyz_weight_accum[update_filter] += weight_accumulate[update_filter]
    
    @torch.no_grad()
    def update_env_gs(self,iter,opt,scene,render_pkg):
        env_densify_untile_iter = 24000
        env_densify_inter = 500
        env_opacity_reset_inter = 6000
        env_densify_grad_thres = 1e-4/2
        env_min_gradient = None
        env_densify_size_threshold = 0.01 # percent dense
        env_max_scene_threshold = 0.1
        env_min_opacity = 0.05

        size_threshold = 20 if iter > env_opacity_reset_inter else None
        screen_threshold = None#0.1 if iter < 8000 else None
        
        env_min_weight_threshold = 0.1
        env_prune_visibility = True

        # if iter == 24000:
            # return
            # self.freeze_geo()
            # return 

        self.update_learning_rate(iter-self.start_iter)
        if iter > 0 and iter % 1000 == 0:
            self.oneupSHdegree()

        if iter >= 21000:
            return
        
        visibility_filter = render_pkg["visibility_filter"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        acc_weights = render_pkg["weight_accumulate"]
        if iter > 0 and iter < env_densify_untile_iter:
            self.add_densification_stats(viewspace_point_tensor,visibility_filter,acc_weights)
        if iter > 0 and iter < env_densify_untile_iter and iter % env_densify_inter == 0:
            print("Before Densify f{:06d}: {:06d} points".format(iter, self.get_xyz.shape[0]))
            self.densify_and_prune(
                max_grad=env_densify_grad_thres,
                min_opacity=env_min_opacity,
                extent=scene.cameras_extent,
                max_screen_size=size_threshold,
                split_screen_threshold=screen_threshold
            )
            print("After Densify f{:06d}: {:06d} points".format(iter, self.get_xyz.shape[0]))

            # if iter % env_opacity_reset_inter == 0:
            #     self.reset_opacity()


    def get_normal(self, scaling_modifier, dir_pp_normalized, return_delta=False): 
        splat2world = self.get_covariance(scaling_modifier)
        normals_raw = splat2world[:,2,:3] 
        normals_raw, positive = flip_align_view(normals_raw, dir_pp_normalized)

        if return_delta:
            delta_normal1 = self._normal1 
            delta_normal2 = self._normal2 
            delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) 
            idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) 
            delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) 
            normals = delta_normal + normals_raw
            normals = safe_normalize(normals) 
            return normals, delta_normal
        else:
            normals = safe_normalize(normals_raw)
            return normals
    
    @torch.no_grad()
    def restore_from_refgs(self,model_args,opt,anchored_lst=[]):  
        # model_args_ = []
        # for _ in model_args:
        #     if isinstance(_, torch.Tensor):
        #         if not _.requires_grad:
        #             model_args_.append(_.detach().clone())
        #         else:
        #             model_args_.append(_.detach().clone().requires_grad_(True))
        #     elif isinstance(_, nn.Parameter):
        #         model_args_.append(nn.Parameter(_.detach().clone().requires_grad_(True)))
        #     else:
        #         model_args_.append(_)

        (self.active_sh_degree, 
        self._xyz, 
        _refl_strength,  
        self._metalness, 
        _roughness, 
        _ori_color, 
        _diffuse_color,
        self._features_dc, 
        self._features_rest,
        _indirect_dc, 
        _indirect_rest,
        _indirect_asg,
        self._scaling, 
        self._rotation, 
        self._opacity,
        _normal1,  
        _normal2,  
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(opt,anchored_lst=anchored_lst)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = xyz_gradient_accum
        self.start_iter = 12500
        # self.active_sh_degree = 0