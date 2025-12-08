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
from torch.autograd import Variable
from math import exp
from kornia.filters import spatial_gradient
from .image_utils import psnr
from utils.image_utils import erode
from utils.lap_loss import LapLoss
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

net_map = None 
laploss_map = None

def lpips_loss(x: torch.Tensor, y: torch.Tensor, net='vgg'):
    global net_map
    if net_map is None:
        import lpips as lpips_module
        print(f'Initializing LPIPS network: {(net)}')
        net_map = lpips_module.LPIPS(net=net, verbose=False).cuda()
        # net_map = lpips_module.LPIPS(net=net, verbose=False, spatial=True).cuda()
    return net_map(x.cuda() * 2. - 1., y.cuda() * 2. - 1.).mean()

def lap_loss(x:torch.Tensor, y:torch.Tensor):
    global laploss_map
    if laploss_map is None:
        laploss_map = LapLoss().cuda()
    return laploss_map(2.*x-1.,2.*y-1.)

def compute_gradients(image):
    sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32).cuda()
    sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32).cuda()
    
    if image.dim() == 4:
        sobel_x = sobel_x.unsqueeze(0).repeat(image.size(1), 1, 1, 1)
        sobel_y = sobel_y.unsqueeze(0).repeat(image.size(1), 1, 1, 1)
    
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.size(1))
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.size(1))
    return grad_x, grad_y

def gradient_loss_fn(x, y, bg_mask:torch.Tensor=None):
    # gen_grad_x, gen_grad_y = compute_gradients(x)
    # gt_grad_x, gt_grad_y = compute_gradients(y)
    # loss_x = F.l1_loss(gen_grad_x, gt_grad_x, reduction='mean')
    # loss_y = F.l1_loss(gen_grad_y, gt_grad_y, reduction='mean')
    grad_x = get_img_grad_weight(x.squeeze())
    grad_y = get_img_grad_weight(y.squeeze())
    if bg_mask is not None:
        return F.l1_loss(grad_x[bg_mask.bool().view_as(grad_x)],grad_y[bg_mask.bool().view_as(grad_y)],reduction='sum') / bg_mask.sum()
    else:
        return F.l1_loss(grad_x,grad_y,reduction='mean')

def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def smooth_loss(data):
    return spatial_gradient(data[None],order=1)[0].abs().sum(1).mean()

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img


def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration, image_weight=None, bg_mask=None):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    visibility_filter = render_pkg["visibility_filter"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()

    Ll1 = l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    loss = torch.zeros_like(loss0)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += loss0

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        # loss_normal_render_depth = ((1 - (rendered_normal * surf_normal).sum(dim=0)) * image_weight)[None]
        if image_weight is not None:
            loss_normal_render_depth = (image_weight * (((surf_normal - rendered_normal)).abs().sum(0))).mean()
        else:
            loss_normal_render_depth = ((1 - (rendered_normal * surf_normal).sum(dim=0)))[None]
            loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_smooth_from_iter and iteration < opt.normal_smooth_until_iter:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)

    # if opt.use_albedo_smoothness  and opt.lambda_albedo_smoothness > 0 and iteration > opt.albedo_smoothness_start_iter:
    #     loss_albedo_smoothness = smooth_loss(rendered_image)
    #     tb_dict["loss_albedo_smoothness"] = loss_albedo_smoothness.item()
    #     lambda_albedo_smoothness = opt.lambda_albedo_smoothness
    #     loss = loss + lambda_albedo_smoothness * loss_albedo_smoothness
    # else:
    #     tb_dict["loss_albedo_smoothness"] = torch.zeros_like(loss)

    if opt.use_perceptual_loss and iteration > opt.perceptual_loss_start_iter:
        perc_loss = lpips_loss(rendered_image.unsqueeze(0),gt_image.unsqueeze(0))
        loss = loss + opt.lambda_perceptual_loss * perc_loss
        tb_dict["perceptual_loss"] = perc_loss.detach()
    
    # if opt.use_laploss and iteration > opt.laploss_start_iter:
    #     # import ipdb;ipdb.set_trace()
    #     # laploss = lap_loss(rendered_image.unsqueeze(0),gt_image.unsqueeze(0))
    #     # loss = loss + opt.lambda_laploss * laploss
    #     # tb_dict["lap_loss"] = laploss.detach() 
    
    #     gradient_loss = gradient_loss_fn(rendered_image.unsqueeze(0),gt_image.unsqueeze(0),bg_mask=bg_mask)
    #     loss = loss + gradient_loss

    #     if iteration % 200 == 0:
    #         # print(laploss.detach() * opt.lambda_laploss)
    #         print(gradient_loss.detach())
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask