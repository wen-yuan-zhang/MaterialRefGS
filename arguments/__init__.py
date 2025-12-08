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


from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    if value:  
                        group.add_argument("--" + key, ("-" + key[0:1]), dest=key, default=value, action="store_true")
                        group.add_argument("--no-" + key, dest=key, action="store_false")
                    else:  
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if value:  
                        group.add_argument("--" + key, dest=key, default=value, action="store_true")
                        group.add_argument("--no-" + key, dest=key, action="store_false")
                    else:  
                        group.add_argument("--" + key, default=value, action="store_true")
                elif t == list: # #
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # Rendering Settings
        self.sh_degree = 3
        self._resolution = -1
        self._white_background = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        self.multi_view_num = 8
        self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01
        self.multi_view_max_dis = 1.5
        self.ncc_scale = 1.0
        
        # Paths
        self._source_path = ""
        self._model_path = ""
        self._images = "images"

        # Device Settings
        self.data_device = "cuda"
        self.eval = False

        # EnvLight Settings
        self.envmap_max_res = 128
        self.envmap_max_roughness = 0.5
        self.envmap_min_roughness = 0.08
        self.relight = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # Processing Settings
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.use_asg = False

        # Debugging
        self.depth_ratio = 0.0
        self.debug = False

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Learning Rate Settings
        self.iterations = 50_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.features_lr = 0.0075 
        self.indirect_lr = 0.0075 
        self.asg_lr = 0.0075 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001


        self.ori_color_lr = 0.0075 
        self.refl_strength_lr =  0.005 
        self.roughness_lr =  0.005 
        self.metalness_lr = 0.01
        self.normal_lr = 0.006
        self.indirect_ceof_lr = 0.005

        self.envmap_cubemap_lr = 0.01
        
        # Densification Settings
        self.percent_dense = 0.01

        # Regularization Parameters
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal_render_depth = 0.05
        self.lambda_normal_smooth = 0.0
        self.lambda_depth_smooth = 0.0
        self.wo_image_weight = False


        # initial values
        self.init_roughness_value = 0.1
        self.init_refl_value = 0.1
        self.init_refl_value_vol = 0.01
        self.rough_msk_thr = 0.01
        self.refl_msk_thr = 0.02
        self.refl_msk_thr_vol = 0.02

        self.enlarge_scale = 1.5


        # Opacity and Densify Settings
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25000 

        # # Extra settings
        self.densify_grad_threshold = 0.0002
        self.prune_opacity_threshold = 0.05



        # Stage Settings
        self.initial = 0
        self.init_until_iter = 0 
        self.volume_render_until_iter = 18000 
        self.normal_smooth_from_iter = 0
        self.normal_smooth_until_iter = 18000
                
        self.indirect = 0
        self.indirect_from_iter =  20000 

        self.feature_rest_from_iter = 5_000
        self.normal_prop_until_iter = 25_000 

        self.normal_prop_interval = 1000
        self.opac_lr0_interval = 200
        self.densification_interval_when_prop = 500



        self.normal_loss_start = 0
        self.dist_loss_start = 3000

        # Environmental Scoping
        self.use_env_scope = False
        self.env_scope_center = [0., 0., 0.]
        self.env_scope_radius = 0.0
        
        # SRGB Transformation
        self.srgb = False

        # mesh
        self.voxel_size = -1.0
        self.depth_trunc = -1.0
        self.sdf_trunc = -1.0
        self.mesh_res = 512
        self.num_cluster = 1

        self.wo_image_weight = False
        # self.single_view_weight = 0.015
        # self.single_view_weight_from_iter = 7000

        self.use_virtul_cam = False
        self.virtul_cam_prob = 0.5
        self.use_multi_view_trim = True
        self.multi_view_ncc_weight = 0.15
        self.multi_view_geo_weight = 0.03
        self.multi_view_weight_from_iter = 10000
        self.multi_view_patch_size = 3
        self.multi_view_sample_num = 102400
        self.multi_view_pixel_noise_th = 1.0
        self.wo_use_geo_occ_aware = False

        self.use_raytracing_indirect_light = False
        self.use_perceptual_loss = True
        self.lambda_perceptual_loss = 0.1
        self.perceptual_loss_start_iter = 18000 # (21000)
        
        self.use_laploss = False
        self.lambda_laploss = 0.0000005
        self.laploss_start_iter = 10400
        
        self.rghmtl_warp_loss_start_iter = 10000
        self.use_metallic_warp_loss = True
        self.use_roughness_warp_loss = True
        self.metallic_warp_weight = 0.05
        self.roughness_warp_weight = 0.05   
        self.use_backgroud_mask = True
        self.directional_rghmtl_warp_alignment = True
        
        self.dilate_size = 7
        self.edge_aware_in_warp = True

        self.ref_score_start_iter = 10000
        self.ref_score_loss_weight = 0.01
        self.ref_score_loss_inv_weight = 0.005

        self.use_albedo_smoothness = False
        self.lambda_albedo_smoothness = 0.015
        self.albedo_smoothness_start_iter = 10000

        self.tel_thres = 0.1

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
