import torch
import imageio
import numpy as np
from . import renderutils as ru
from .light_utils import *
import nvdiffrast.torch as dr
import imageio
import numpy as np


def linear_to_srgb(linear):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.clip(linear,1e-4,255) ** (5 / 12) - 11) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class EnvLight(torch.nn.Module):

    def __init__(self, path=None, device=None, scale=1.0, min_res=16, max_res=128, min_roughness=0.08, max_roughness=0.5, trainable=False):
        super().__init__()
        self.device = device if device is not None else 'cuda' # only supports cuda
        self.scale = scale # scale of the hdr values
        self.min_res = min_res # minimum resolution for mip-map
        self.max_res = max_res # maximum resolution for mip-map
        self.min_roughness = min_roughness
        self.max_roughness = max_roughness
        self.trainable = trainable

        # init an empty cubemap
        self.base = torch.nn.Parameter(
            torch.zeros(6, self.max_res, self.max_res, 3, dtype=torch.float32, device=self.device),
            requires_grad=self.trainable,
        )
        
        # try to load from file (.hdr or .exr)
        if path is not None:
            self.load(path)
        
        self.build_mips()


    def load(self, path):
        """
        Load an .hdr or .exr environment light map file and convert it to cubemap.
        """
        # # load latlong env map from file
        # image = imageio.imread(path)  # Load .hdr file
        # if image.dtype != np.float32:
        #     image = image.astype(np.float32) / 255.0  # Scale to [0,1] if not already in float
        # 从文件中加载图像
        hdr_image = imageio.imread(path)
        
        if hdr_image.dtype != np.float32:
            raise ValueError("HDR image should be in float32 format.")

        ldr_image = linear_to_srgb(hdr_image)
        # 确保图像为浮点类型
        image = torch.from_numpy(ldr_image).to(self.device) *  self.scale
        image = torch.clamp(image, 0.001 , 1-0.001)
        image = inverse_sigmoid(image)

        # Convert from latlong to cubemap format
        cubemap = latlong_to_cubemap(image, [self.max_res, self.max_res], self.device)

        # Assign the cubemap to the base parameter
        self.base.data = cubemap 

    def build_mips(self, cutoff=0.99):
        """
        Build mip-maps for specular reflection based on cubemap.
        """
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.min_res:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.max_roughness - self.min_roughness) + self.min_roughness
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def get_mip(self, roughness):
        """
        Map roughness to mip level.
        """
        return torch.where(
            roughness < self.max_roughness, 
            (torch.clamp(roughness, self.min_roughness, self.max_roughness) - self.min_roughness) / (self.max_roughness - self.min_roughness) * (len(self.specular) - 2), 
            (torch.clamp(roughness, self.max_roughness, 1.0) - self.max_roughness) / (1.0 - self.max_roughness) + len(self.specular) - 2
        )
        

    def __call__(self, l, mode=None, roughness=None):
        """
        Query the environment light based on direction and roughness.
        """
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            l = l.reshape(1, 1, -1, l.shape[-1])
            if roughness is not None:
                roughness = roughness.reshape(1, 1, -1, 1)

        if mode == "diffuse":
            # Diffuse lighting
            light = dr.texture(self.diffuse[None, ...], l, filter_mode='linear', boundary_mode='cube')
        elif mode == "pure_env":
            # Pure environment light (no mip-map)
            light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        else:
            # Specular lighting with mip-mapping
            miplevel = self.get_mip(roughness)
            light = dr.texture(
                self.specular[0][None, ...], 
                l,
                mip=list(m[None, ...] for m in self.specular[1:]), 
                mip_level_bias=miplevel[..., 0], 
                filter_mode='linear-mipmap-linear', 
                boundary_mode='cube'
            )

        light = light.view(*prefix, -1)
        
        return torch.sigmoid(light)
