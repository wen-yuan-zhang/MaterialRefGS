from pointnet2_ops.pointnet2_utils import ball_query,furthest_point_sample
from typing import List, Tuple, Optional, Literal
from collections import defaultdict
import torch



def get_points_from_depth(self, fov_camera, depth, scale=1):
    st = int(max(int(scale/2)-1,0))
    depth_view = depth.squeeze()[st::scale,st::scale]
    rays_d = fov_camera.get_rays(scale=scale)
    depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
    pts = (rays_d * depth_view[..., None]).reshape(-1,3)
    # R = torch.tensor(fov_camera.R).float().cuda()
    # T = torch.tensor(fov_camera.T).float().cuda()
    R = fov_camera.R.float().cuda()
    T = fov_camera.T.float().cuda()
    pts = (pts-T)@R.transpose(-1,-2)
    return pts

def ref_scores_max_pooling(ref_score_images,viewpoint_cam_lst,scene_extend,radius_ratio=0.1,n_samples=7*7,backend=Literal['ball query','knn'],calc_depth_points_radius=False):
    '''
    ref_score_images: dict of (N,H,W)
    viewpoint_cam_lst: N Camera
    scene_extend: (1,) radius of camera extend
    radius_ratio: (1,) radius ratio of ball query
    max_neighbors: (1,) max number of neighbors for max pooling    
    '''
    H,W,_ = viewpoint_cam_lst[0].HWK
    N = len(viewpoint_cam_lst)


    if backend != 'ball query':
        raise NotImplementedError("Only ball query is supported")
    
    ref_score_images_pooled = defaultdict()
    total_points = []
    ref_score_lst = []
    for i,viewpoint_cam in enumerate(viewpoint_cam_lst):
        ref_score_image = ref_score_images[i]
        pts = get_points_from_depth(viewpoint_cam,ref_score_image)
        total_points.append(pts)
        ref_score_lst.append(ref_score_image.reshape(-1))
    
    total_points = torch.cat(total_points,dim=0).unsqueeze_(0) # (1,-1,3)
    ref_score = torch.cat(ref_score_lst,dim=0).unsqueeze_(0) # (1,-1)

    # radius of depth points # TODO check radius setup method
    if calc_depth_points_radius:
        depth_points_radius = furthest_point_sample(total_points,2).squeeze()
        depth_points_radius = (total_points[depth_points_radius[0]]-total_points[depth_points_radius[1]]).norm(dim=-1)
        radius = depth_points_radius * radius_ratio
    else:
        radius = scene_extend * radius_ratio
    

    # ball query
    neighbor_points_inds = ball_query(radius,n_samples,total_points,total_points).squeeze_() # (-1，n_samples)
    ref_score_neighbors = ref_score[neighbor_points_inds.reshape(-1)].reshape(-1,n_samples).max(dim=-1)[0] # (-1)
    ref_score_images_pooled = ref_score_neighbors.reshape(N,H,W)

    ref_score_images_pooled_dict = defaultdict()
    for idx,viewpoint_cam in enumerate(viewpoint_cam_lst):
        image_name = viewpoint_cam.image_name
        ref_score_images_pooled_dict[image_name] = ref_score_images_pooled[idx]
    
    return ref_score_images_pooled_dict

def ref_scores_max_pooling_wiht_mask(ref_score_images,mask_images,viewpoint_cam_lst,scene_extend,radius_ratio=0.1,n_samples=7*7,backend=Literal['ball query','knn'],calc_depth_points_radius=False):
    '''
    ref_score_images: dict of (N,H,W)
    viewpoint_cam_lst: N Camera
    scene_extend: (1,) radius of camera extend
    radius_ratio: (1,) radius ratio of ball query
    max_neighbors: (1,) max number of neighbors for max pooling    
    '''
    H,W,_ = viewpoint_cam_lst[0].HWK
    N = len(viewpoint_cam_lst)


    if backend != 'ball query':
        raise NotImplementedError("Only ball query is supported")
    
    ref_score_images_pooled = defaultdict()
    total_points = []
    ref_score_lst = []
    length = []
    for i,viewpoint_cam in enumerate(viewpoint_cam_lst):
        ref_score_image = ref_score_images[i]
        mask = mask_images[viewpoint_cam.image_name].resshape(-1)
        pts = get_points_from_depth(viewpoint_cam,ref_score_image)
        total_points.append(pts[mask_images[mask]])
        ref_score_lst.append(ref_score_image.reshape(-1)[mask])
        length.append(mask.sum())
    
    total_points = torch.cat(total_points,dim=0).unsqueeze_(0) # (1,-1,3)
    ref_score = torch.cat(ref_score_lst,dim=0).unsqueeze_(0) # (1,-1)

    # radius of depth points # TODO check radius setup method
    if calc_depth_points_radius:
        depth_points_radius = furthest_point_sample(total_points,2).squeeze()
        depth_points_radius = (total_points[depth_points_radius[0]]-total_points[depth_points_radius[1]]).norm(dim=-1)
        radius = depth_points_radius * radius_ratio
    else:
        radius = scene_extend * radius_ratio
    

    # ball query
    neighbor_points_inds = ball_query(radius,n_samples,total_points,total_points).squeeze_() # (-1，n_samples)
    ref_score_neighbors = ref_score[neighbor_points_inds.reshape(-1)].reshape(-1,n_samples).max(dim=-1)[0] # (-1)
    ref_score_images_pooled = ref_score_neighbors.split(length,dim=0)

    ref_score_images_pooled_dict = defaultdict()
    for idx,viewpoint_cam in enumerate(viewpoint_cam_lst):
        mask = mask_images[viewpoint_cam.image_name].reshape(-1)
        image_name = viewpoint_cam.image_name
        ref_score_images_pooled_i = ref_score_images[viewpoint_cam.image_name].reshape(-1)
        ref_score_images_pooled_i[mask] = ref_score_images_pooled[idx]
        ref_score_images_pooled_dict[image_name] = ref_score_images_pooled_i.reshape(H,W)
    
    return ref_score_images_pooled_dict