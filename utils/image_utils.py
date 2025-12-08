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
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import numpy as np
import cv2


def psnr(img1, img2):
    # mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    mse = (((img1 - img2) ** 2).reshape(img1.shape[0], -1)).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image

def visualize_d_mask(d_mask,H,W):
    d_mask_show = (d_mask.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
    d_mask_show_color = torch.from_numpy(d_mask_show_color).float().cuda().permute(2, 0, 1)/255.
    d_mask_show_color =  d_mask_show_color[[2,1,0],...]
    return d_mask_show_color


def visualize_depth(depth, near=0.2, far=13, norm=True):
    depth = depth.detach().cpu().squeeze().numpy()
    if norm : depth = (depth-depth.min())/(1e-20+depth.max()-depth.min())
    depth = (depth * 255).clip(0,255).astype(np.uint8)
    depth_c = cv2.applyColorMap(depth,cv2.COLORMAP_JET)
    depth_c = torch.from_numpy(depth_c).float().cuda().permute(2, 0, 1)/255.
    depth_c =  depth_c[[2,1,0],...]
    return depth_c


def visualize_depth2(depth, near=0.2, far=13, norm=True):
    depth = depth.detach().cpu().squeeze().numpy()
    # if norm : depth = (depth-depth.min())/(1e-20+depth.max()-depth.min())
    std = depth.std()
    depth = depth.clip(0,6*std)
    depth = (depth-depth.min())/(1e-20+depth.max()-depth.min())
    depth = (depth * 255).clip(0,255).astype(np.uint8)
    depth_c = cv2.applyColorMap(depth,cv2.COLORMAP_JET)
    depth_c = torch.from_numpy(depth_c).float().cuda().permute(2, 0, 1)/255.
    depth_c =  depth_c[[2,1,0],...]
    return depth_c

def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out

def dilate(img_in, dilate_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    img_out = cv2.dilate(img_out, kernel, iterations=1)

    return img_out

def dilated_edges_imgs(img_in, dilate_size=4, thres1=0., thres2=80.):
    img_out = img_in.detach().cpu().clone().permute((1,2,0)).numpy()
    img_out = (img_out * 255).astype(np.uint8)
    img_out = cv2.cvtColor(img_out,cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_out,thres1,thres2)
    img_out = dilate(edges,dilate_size=dilate_size)
    img_out = torch.from_numpy(img_out/255.).to(img_in.device)
    return img_out