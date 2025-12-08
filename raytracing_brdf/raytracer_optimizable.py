
import numpy as np
import torch

# CUDA extension
import _raytracing_brdf as _backend
import torch.utils
from scene.light import EnvLight
from plyfile import PlyData,PlyElement
from utils.refl_utils import reflection,safe_normalize
from utils.graphics_utils import rotation_between_z
import nvdiffrast.torch as dr
FG_LUT = torch.from_numpy(np.fromfile("./assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)).cuda()

from ipdb import set_trace as st
import torch.nn as nn
import torch.optim as optim 
from scene.gaussian_model import inverse_sigmoid
from torch.nn.functional import sigmoid

class RayTracer():
    def __init__(self, vertices, triangles, ply_path=None):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]
        if ply_path is not None:
            vertices, triangles, vertex_colors, vertex_attrs = self.load_from_ply_file(ply_path)
        else:
            if torch.is_tensor(vertices): vertices = vertices.detach().cpu().numpy()
            if torch.is_tensor(triangles): triangles = triangles.detach().cpu().numpy()

        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."

        # implementation
        self.impl = _backend.create_raytracer(vertices, triangles)
        if ply_path is not None:
            self.mappings = self.impl.get_triangels_ids_mapping().contiguous().cuda()
            self.vertices = torch.from_numpy(vertices).cuda().contiguous()
            self.triangles = torch.from_numpy(triangles).cuda().contiguous()
            self.vertex_colors = vertex_colors
            self.vertex_attrs = vertex_attrs

            for k_,v_ in self.vertex_attrs.items():
                if k_ in ["albedo"]:
                    self.vertex_attrs[k_] = nn.Parameter(inverse_sigmoid(v_))

    def get_optimizer(self,learning_rate=1e-6):
        params = [self.vertex_attrs[k_] for k_ in ["albedo"]]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        return self.optimizer

    def trans_ids(self, face_ids:torch.Tensor):
        face_ids[face_ids>=0] = self.mappings[face_ids[face_ids>=0]]
        return face_ids

    def load_from_ply_file(self,ply_path):
        ply_data = PlyData.read(ply_path)  
        if 'vertex' not in ply_data:  
            raise ValueError("PLY file does not contain 'vertex' element!")  
        
        vertex_data = ply_data['vertex']  
        dtype = vertex_data.data.dtype  
        vertex_array = np.array(vertex_data.data) # NOTE
        vertex_array = np.column_stack([vertex_array[field] for field in vertex_array.dtype.names])
        field_names = dtype.names  

        field_indices = {}  
        for i, name in enumerate(field_names):  
            prefix, *postfix = name.split('_')  
            if prefix not in field_indices:  
                field_indices[prefix] = []  
            field_indices[prefix].append(i)  

        xyz_indices = [field_names.index('x'), field_names.index('y'), field_names.index('z')]  
        vertices = vertex_array[:, xyz_indices]  

        rgb_indices = [field_names.index('red'), field_names.index('green'), field_names.index('blue')]  
        vertex_colors = vertex_array[:, rgb_indices]  

        vertex_attrs = {}  
        for prefix, indices in field_indices.items():  
            if len(indices) >= 1 and not (prefix in ["x","y","z","green","red","blue"]):  
                vertex_attrs[prefix] = torch.from_numpy(vertex_array[:, indices]).cuda()  
        faces = None  
        if 'face' in ply_data:  
            face_data = ply_data['face']  
            faces = np.array([list(face[0]) for face in face_data])

        return vertices, faces, vertex_colors, vertex_attrs

    def trace(self, rays_o, rays_d, inplace=False, return_faceids=False):
        # rays_o: torch.Tensor, cuda, float, [N, 3]
        # rays_d: torch.Tensor, cuda, float, [N, 3]
        # inplace: write positions to rays_o, face_normals to rays_d

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        if not inplace:
            # allocate
            positions = torch.empty_like(rays_o)
            face_normals = torch.empty_like(rays_d)
        else:
            positions = rays_o
            face_normals = rays_d

        depth = torch.empty_like(rays_o[:, 0])
        triangle_indices = torch.empty_like(rays_o[:, 0],dtype=torch.int32)
        
        # inplace write intersections back to rays_o
        self.impl.trace(rays_o, rays_d, positions, face_normals, depth, triangle_indices) # [N, 3]

        positions = positions.view(*prefix, 3)
        face_normals = face_normals.view(*prefix, 3)
        depth = depth.view(*prefix)
        triangle_indices = triangle_indices.view(*prefix)
        triangle_indices = self.trans_ids(triangle_indices) #IMPORTACE

        if not return_faceids:
            return positions, face_normals, depth
        else:
            return positions, face_normals, depth, triangle_indices

    @staticmethod
    def ggx_distribution(n_dot_h, roughness):  
        alpha = roughness ** 2  
        alpha = torch.clamp(roughness ** 2, min=0.001)
        numerator = alpha ** 2  
        denominator = np.pi * ((n_dot_h ** 2) * (alpha ** 2 - 1) + 1) ** 2  
        return numerator / denominator  

    @staticmethod
    def ggx_importance_sampling(rougness, v, n, num_sample):
        """
        rougness: (N,1)
        v: viewpoint direction (N,3)
        n: normal (N,3)
        """
        B = v.shape[0]
        alpha = rougness ** 2
        x, y = torch.rand((B,num_sample)).to(v.device), torch.rand((B,num_sample)).to(v.device)
        phi_h = 2 * torch.pi * x
        theta_h = torch.arccos(torch.sqrt((1 - y) / (1 + (alpha ** 2 - 1) * y))) 
        
        sin_theta_h = torch.sin(theta_h)
        cos_theta_h = torch.cos(theta_h) # (B,N)

        h_local = torch.stack([
            sin_theta_h * torch.cos(phi_h),
            sin_theta_h * torch.sin(phi_h),
            cos_theta_h
        ], dim=-2) # (B,3,N) local half vector

        rotation_matrix = rotation_between_z(n) # (B,3,3)
        h_global = (rotation_matrix @ h_local) # (B,3,N)
        h_global = torch.nn.functional.normalize(h_global,dim=-2).transpose(-1,-2) # (B,N,3)
        # st()
        a2 = alpha**2
        d = (cos_theta_h * a2 - sin_theta_h) * cos_theta_h + 1
        D = a2 / (torch.pi*d*d)
        pdf_h = torch.clamp(D * cos_theta_h, min=1e-6) # (B,N)
        # Jacobian Determinant 
        v_dot_h = (v.unsqueeze(1).expand_as(h_global)*h_global).sum(-1,keepdim=True) #(B,N,1)
        v_dot_h = torch.clamp(v_dot_h, min=1e-2)
        incident_dirs = 2.0 * v_dot_h * h_global - v.unsqueeze(1).expand_as(h_global)
        pdf_l = pdf_h / (4. * v_dot_h.squeeze())

        return h_global, incident_dirs ,pdf_l.clamp(1e-6)
    
    # Fresnel-Schlick approximation  
    @staticmethod
    def fresnel_schlick(cos_theta, f0):  
        return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5  

    # GGX Geometry term (Smith's method)  
    @staticmethod
    def geometry_schlick_ggx(n_dot_x, roughness):  
        alpha = roughness ** 2  
        k = (alpha + 1) ** 2 / 8  
        return n_dot_x / (n_dot_x * (1 - k) + k)  

    @staticmethod
    def geometry_smith(n_dot_v, n_dot_l, roughness):  
        g_v = RayTracer.geometry_schlick_ggx(n_dot_v, roughness)  
        g_l = RayTracer.geometry_schlick_ggx(n_dot_l, roughness)  
        return g_v * g_l  

    # Cook-Torrance BRDF  
    @staticmethod
    def cook_torrance_brdf(v, l, n, roughness, f0):  
        h = safe_normalize(v + l) 

        n_dot_v = torch.clamp(torch.einsum("ij,ij->i",n, v), 1e-4,1)[:,None]  
        n_dot_l = torch.clamp(torch.einsum("ij,ij->i",n, l), 1e-4,1)[:,None]   
        n_dot_h = torch.clamp(torch.einsum("ij,ij->i",n, h), 1e-4,1)[:,None]   
        v_dot_h = torch.clamp(torch.einsum("ij,ij->i",v, h), 1e-4,1)[:,None]    

        F = RayTracer.fresnel_schlick(v_dot_h, f0)  
        G = RayTracer.geometry_smith(n_dot_v, n_dot_l, roughness)  
        D = RayTracer.ggx_distribution(n_dot_h, roughness)  
        mask = torch.bitwise_and((n_dot_v>1e-4),(n_dot_l>1e-4))
        mask = torch.bitwise_and(mask,(n_dot_h>1e-4))
        mask = torch.bitwise_and(mask,(v_dot_h>1e-4)).squeeze()
        brdf = torch.zeros_like(v).to(v.device) # (N,3)
        brdf[mask] = ((F * G * D) / (4 * n_dot_v * n_dot_l+1e-8))[mask]
        return brdf

    def barycentric_interpolation(self,p,fp,fv):
        def area(v1, v2, v3):
            return 0.5 * torch.abs(v1[...,0] * (v2[...,1] - v3[...,1]) +  
                        v2[...,0] * (v3[...,1] - v1[...,1]) +  
                        v3[...,0] * (v1[...,1] - v2[...,1])) 
        a,b,c=fp[...,0,:],fp[...,1,:],fp[...,2,:]
        fa,fb,fc=fv[...,0,:],fv[...,1,:],fv[...,2,:]
        return (area(p,b,c).unsqueeze(-1) * fa + area(p,c,a).unsqueeze(-1) * fb + area(p,a,b).unsqueeze(-1) * fc) / area(a,b,c).unsqueeze(-1)

    def secondary_indirect_color(self, surface_pos:torch.Tensor, rays_v:torch.Tensor,face_normals:torch.Tensor, triangle_indices: torch.Tensor, hit_depth:torch.Tensor, env_map:EnvLight):
        B = surface_pos.shape[0]
        assert ((hit_depth==10.)==(triangle_indices==-1)).all()

        # vertex_normal = self.vertex_attrs["normal"][self.triangles[triangle_indices].reshape(-1)].reshape(-1,3,3).float()
        # vertex_normal = vertex_normal * 2 - 1.
        # vertex_pos_tot = self.vertices[self.triangles[triangle_indices].reshape(-1)].reshape(-1,3,3).float()
        # face_normals = self.barycentric_interpolation(surface_pos,vertex_pos_tot,vertex_normal)


        final_color = torch.zeros((B,3)).to(surface_pos.device)
        invisible_indices = (hit_depth<10.)
        if ((~invisible_indices).sum()>0):
            final_color[~invisible_indices] = env_map(-rays_v[~invisible_indices],mode="pure_env")

        surface_pos_inv = surface_pos[invisible_indices]
        rays_v_inv = rays_v[invisible_indices]
        face_normals_inv = face_normals[invisible_indices]
        triangle_indices_inv = triangle_indices[invisible_indices]

        vertex_ids = self.triangles[triangle_indices_inv].reshape(-1) # (N,3)
        vertex_pos = self.vertices[vertex_ids].reshape((-1,3,3)).float() # (N,3,C)
        vertex_diffuse = self.vertex_attrs["diffuse"][vertex_ids].reshape((-1,3,3)).float()
        vertex_roughness = self.vertex_attrs["roughness"][vertex_ids].reshape((-1,3,1)).float()
        vertex_albedo = sigmoid(self.vertex_attrs["albedo"][vertex_ids].reshape((-1,3,3)).float())
        vertex_metallic = self.vertex_attrs["metallic"][vertex_ids].reshape((-1,3,1)).float()
        vertex_normal = self.vertex_attrs["normal"][vertex_ids].reshape((-1,3,3)).float()
        vertex_normal = vertex_normal * 2. - 1.

        diffuse = self.barycentric_interpolation(surface_pos_inv,vertex_pos,vertex_diffuse).float().clamp(0,1) #(N,3)
        refl_strength = self.barycentric_interpolation(surface_pos_inv,vertex_pos,vertex_metallic).clamp(0,1) #(N,1)
        roughness = self.barycentric_interpolation(surface_pos_inv,vertex_pos,vertex_roughness).clamp(0,1) #(N,1)
        albedo = self.barycentric_interpolation(surface_pos_inv,vertex_pos,vertex_albedo).clamp(0,1) #(N,3)
        normal_map = self.barycentric_interpolation(surface_pos_inv,vertex_pos,vertex_normal).clamp(-1,1)

        rays_l,NdotV = reflection(rays_v_inv,normal_map) # NOTE face_normal是有向的吗?
        rays_l = safe_normalize(rays_l)

        ## secondary brdf part
        global FG_LUT
        fg_uv = torch.cat([NdotV,roughness],-1).clamp(0,1).float()
        fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape((-1,2))

        # Compute direct light
        with torch.no_grad():
            direct_light = env_map(rays_l,roughness=roughness)

        specular_weight = ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2])
        specluar_light = specular_weight * direct_light

        final_color[invisible_indices] = (1-refl_strength) * diffuse + specluar_light

        # st()

        return final_color
    
    def get_materials_by_faceids(self,surf_points:torch.Tensor,face_ids:torch.Tensor): # (N,3), (N)

        vertex_ids = self.triangles[face_ids].reshape(-1) # (N,3)
        vertex_pos = self.vertices[vertex_ids].reshape((-1,3,3)).float() # (N,3,C)
        vertex_diffuse = self.vertex_attrs["diffuse"][vertex_ids].reshape((-1,3,3)).float()
        vertex_roughness = self.vertex_attrs["roughness"][vertex_ids].reshape((-1,3,1)).float()
        vertex_albedo = sigmoid(self.vertex_attrs["albedo"][vertex_ids].reshape((-1,3,3)).float())
        vertex_metallic = self.vertex_attrs["metallic"][vertex_ids].reshape((-1,3,1)).float()
        vertex_normal = self.vertex_attrs["normal"][vertex_ids].reshape((-1,3,3)).float()
        vertex_normal = vertex_normal * 2. - 1.

        diffuse = self.barycentric_interpolation(surf_points,vertex_pos,vertex_diffuse).float().clamp(0,1) #(N,3)
        refl_strength = self.barycentric_interpolation(surf_points,vertex_pos,vertex_metallic).clamp(0,1) #(N,1)
        roughness = self.barycentric_interpolation(surf_points,vertex_pos,vertex_roughness).clamp(0,1) #(N,1)
        albedo = self.barycentric_interpolation(surf_points,vertex_pos,vertex_albedo).clamp(0,1) #(N,3)
        normal_map = self.barycentric_interpolation(surf_points,vertex_pos,vertex_normal).clamp(-1,1)

        return diffuse, refl_strength, roughness, albedo, normal_map


    def shade(self, surface_pos:torch.Tensor, rays_n:torch.Tensor, rays_v:torch.Tensor, roughness:torch.Tensor, metallic:torch.Tensor, albedo:torch.Tensor ,num_samples:int,envmap:EnvLight):
        """
        surface_pos: torch.Tensor, cuda, float [N,3] | surface position
        rays_n: torch.Tensor, cuda, float [N,3] | surface normal 
        rays_v: torch.Tensor, cuda, float [N,3] | view direction (off the surface)
        roughness: torch.Tensor, cuda, float [N,1] | roughness
        """ # NOTE rays_d direction defination
        # N = surface_pos.shape[0]
        # h_global, incident_dirs, pdf_l = self.ggx_importance_sampling(roughness,rays_v,rays_n,num_samples)
        
        incident_dirs,_ = reflection(rays_v,rays_n)
        safe_normalize(incident_dirs)

        hit_pos, hit_normal, hit_depth, triangle_indices = self.trace(
            surface_pos,incident_dirs,return_faceids=True
        ) 

        # hit_pos, hit_normal, hit_depth, triangle_indices = self.trace(
        #     surface_pos.unsqueeze(1).expand_as(incident_dirs).reshape(-1,3),
        #     incident_dirs.reshape(-1,3)
        # ) 

        indirect = self.secondary_indirect_color(hit_pos,-incident_dirs.reshape(-1,3),hit_normal,triangle_indices,hit_depth,envmap)
        return indirect
        # f0 = (0.04 * (1 - metallic) + albedo * metallic).unsqueeze(1).expand_as(incident_dirs).reshape(-1,3)
        # brdf = self.cook_torrance_brdf(
        #     rays_v.unsqueeze(1).expand_as(incident_dirs).reshape(-1,3),
        #     incident_dirs.reshape(-1,3),
        #     rays_n.unsqueeze(1).expand_as(incident_dirs).reshape(-1,3),
        #     roughness.unsqueeze(1).expand_as(incident_dirs)[...,0:1].reshape(-1,1),
        #     f0)
        
        # n_dot_l = torch.einsum(
        #     "ijk,ijk->ij",
        #     rays_n.unsqueeze(1).expand_as(incident_dirs),
        #     incident_dirs
        # ).reshape(-1,1)

        # # NOTE 
        # mask = torch.bitwise_and(n_dot_l>0,pdf_l.reshape(-1,1)>0).to(surface_pos.device).squeeze()
        # mc_sample = torch.zeros(N*num_samples,3)
        # mc_sample[mask] = (((brdf * indirect * n_dot_l) / (pdf_l.reshape(-1,1) + 1e-8)))[mask] 
        # specular = mc_sample.reshape((N,num_samples,3)).mean(1)

        # return specular.clamp(0.,1.)