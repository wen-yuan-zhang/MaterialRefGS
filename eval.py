import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
# from gaussian_renderer import render_surfel
from gaussian_renderer.envgs_renderer import render_surfel2 as render_surfel
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from torchvision.utils import save_image, make_grid
from gaussian_renderer.optix_utils import HardwareRendering
from scene.env_gaussian_model import EnvGaussianModel
from utils.image_utils import visualize_depth

def render_set(model_path, views, gaussians, pipeline, background, save_ims, opt,render):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, "test", "renders")
        dir_names = ["rgb","gt","normal","surf_normal","surf_depth","diffuse_map","specular_map","albeldo_map","roughness_map","refl_strength_map",]#"direct_light","visibility","indirect_light"]
        for dir_name in dir_names:
            makedirs(os.path.join(render_path,dir_name),exist_ok=True)


    ssims = []
    psnrs = []
    lpipss = []
    render_times = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        t1 = time.time()
        
        rendering = render(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
        render_time = time.time() - t1
        
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, 0:3, :, :]


        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        lpipss.append(lpips(render_color, gt, net_type='vgg').item())
        render_times.append(render_time)

        to_save_lst = [
            render_color,gt,rendering["rend_normal"]*0.5+0.5,rendering["surf_normal"]*0.5+0.5,visualize_depth(rendering["surf_depth"]),rendering["diffuse_map"],rendering["specular_map"],rendering["base_color_map"],rendering["roughness_map"],rendering["refl_strength_map"],rendering["direct_light"],rendering["visibility"],rendering["indirect_light"]
        ]

        # import ipdb;ipdb.set_trace()
        for i,item in enumerate(to_save_lst):
            if len(item.shape) > 3:
                item = item.squeeze(0)
            if item.shape[0]==1:
                item = item.repeat(3,1,1)
            torchvision.utils.save_image(item, os.path.join(render_path,dir_names[i], '{0:05d}.png'.format(idx)))
            
    ssim_v = np.array(ssims).mean()
    psnr_v = np.array(psnrs).mean()
    lpip_v = np.array(lpipss).mean()
    fps = 1.0 / np.array(render_times).mean()
    print('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))
    dump_path = os.path.join(model_path, 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))

def render_set_train(model_path, views, gaussians, pipeline, background, save_ims, opt, render, iteration):
    if save_ims:
        # Create directories to save rendered images
        render_path = os.path.join(model_path, "train", "renders")
        dir_names = ["rgb","gt","normal","surf_normal","surf_depth","diffuse_map","specular_map","albeldo_map","roughness_map","refl_strength_map"]#"direct_light","visibility","indirect_light"]
        for dir_name in dir_names:
            makedirs(os.path.join(render_path,dir_name),exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None  # When evaluating, reflection mask is disabled
        rendering = render(view, gaussians, pipeline, background, srgb=opt.srgb, opt=opt)
 
        render_color = torch.clamp(rendering["render"], 0.0, 1.0)
        render_color = render_color[None]
        gt = torch.clamp(view.original_image, 0.0, 1.0)
        gt = gt[None, :3, :, :]

        to_save_lst = [
            render_color,gt,rendering["rend_normal"]*0.5+0.5,rendering["surf_normal"]*0.5+0.5,visualize_depth(rendering["surf_depth"]),rendering["diffuse_map"],rendering["specular_map"],rendering["base_color_map"],rendering["roughness_map"],rendering["refl_strength_map"],#]rendering["direct_light"],rendering["visibility"],rendering["indirect_light"]
        ]

        for i,item in enumerate(to_save_lst):
            if len(item.shape) > 3:
                item = item.squeeze(0)
            if item.shape[0]==1:
                item = item.repeat(3,1,1)
            torchvision.utils.save_image(item, os.path.join(render_path,dir_names[i], '{0:05d}.png'.format(idx)))

            

from functools import partial
   
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_ims: bool, op, indirect):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        env_gaussians = EnvGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, env_gaussians=env_gaussians)
        print(env_gaussians.get_xyz.shape)
        print(gaussians.get_xyz.shape)
        indirect_renderer = HardwareRendering()
        render = partial(render_surfel,indirect_renderer,env_gaussians)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        if indirect:
            op.indirect = 1
            gaussians.load_mesh_from_ply(dataset.model_path, iteration)

        
        render_set_train(dataset.model_path, scene.getTrainCameras(), gaussians, pipeline, background, save_ims, op,render,iteration)
        render_set(dataset.model_path, scene.getTestCameras(), gaussians, pipeline, background, save_ims, op, render)
        
        env_dict = gaussians.render_env_map()
        grid = [
            env_dict["env1"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env1.png"))
        grid = [
            env_dict["env2"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(grid, os.path.join(dataset.model_path, "env2.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", default=True,action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images, op, True)
