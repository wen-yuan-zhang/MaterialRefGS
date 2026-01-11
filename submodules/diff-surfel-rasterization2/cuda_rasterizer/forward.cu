/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

// 已修改

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
    const float3& p_orig,        // 高斯点的原始位置
    const glm::vec2 scale,       // 高斯点的缩放因子
    float mod,                   // 缩放修正因子
    const glm::vec4 rot,         // 高斯点的旋转四元数
    const float* projmatrix,     // 投影矩阵
    const float* viewmatrix,     // 视图矩阵
    const int W,                 // 图像宽度
    const int H,                 // 图像高度
    glm::mat3 &T,                // 输出的变换矩阵
    float3 &normal               // 输出的法线
) {
    // 计算旋转矩阵
	// 对应 R = [t𝑢, t𝑣, t𝑤]
    glm::mat3 R = quat_to_rotmat(rot);
    // 计算缩放矩阵
    glm::mat3 S = scale_to_mat(scale, mod);
    // 计算旋转和缩放的组合矩阵
    glm::mat3 L = R * S;

    // 将高斯点中心转换到世界坐标系
	// 对应论文中的 H 的124列
    glm::mat3x4 splat2world = glm::mat3x4(
        glm::vec4(L[0], 0.0),
        glm::vec4(L[1], 0.0),
        glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
    );

    // 投影矩阵
    glm::mat4 world2ndc = glm::mat4(
        projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
        projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
        projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
        projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
    );

    // 从NDC坐标到像素坐标的转换矩阵
    glm::mat3x4 ndc2pix = glm::mat3x4(
        glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
        glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
        glm::vec4(0.0, 0.0, 0.0, 1.0)
    );

    // 计算最终的变换矩阵
	// 对应论文中的 (WH)^T
    T = glm::transpose(splat2world) * world2ndc * ndc2pix;

    normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
    glm::mat3 T,                // 输入的变换矩阵
    float cutoff,               // 截止值，用于计算边界框
    float2& point_image,        // 输出的高斯点在图像平面上的中心
    float2 & extent             // 输出的边界框尺寸
) {
    float3 T0 = {T[0][0], T[0][1], T[0][2]}; // 提取变换矩阵的第一行
    float3 T1 = {T[1][0], T[1][1], T[1][2]}; // 提取变换矩阵的第二行
    float3 T3 = {T[2][0], T[2][1], T[2][2]}; // 提取变换矩阵的第三行

    // Compute AABB
    float3 temp_point = {cutoff * cutoff, cutoff * cutoff, -1.0f}; // 临时点，用于计算
    float distance = sumf3(T3 * T3 * temp_point); // 计算距离
    float3 f = (1 / distance) * temp_point; // 计算缩放因子
    if (distance == 0.0) return false; // 如果距离为0，则返回false

    // 计算高斯点在图像平面上的中心
    point_image = {
        sumf3(f * T0 * T3),
        sumf3(f * T1 * T3)
    };

    // 计算边界框尺寸
    float2 temp = {
        sumf3(f * T0 * T0),
        sumf3(f * T1 * T1)
    };
    float2 half_extend = point_image * point_image - temp;
    extent = sqrtf2(maxf2(1e-4, half_extend));
    return true;
}


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif

#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
	float cutoff = 3.0f;
#endif

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(extent.x, extent.y));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,            // 每个瓦片的起始和结束范围
	const uint32_t* __restrict__ point_list,     // 按照瓦片排序的点列表
	const int S, int W, int H,                                // 图像宽度和高度
	float focal_x, float focal_y,                // x和y方向的焦距
	const float2* __restrict__ points_xy_image,  // 高斯点在图像平面中的位置
	const float* __restrict__ colors,          // 高斯点的特征（颜色）
	const float* __restrict__ features,		// //额外特征
	const float* __restrict__ transMats,         // 变换矩阵
	const float* __restrict__ depths,            // 高斯点的深度
	const float4* __restrict__ normal_opacity,   // 高斯点的法线和不透明度
	float* __restrict__ final_T,                 // 累积透明度
	uint32_t* __restrict__ n_contrib,            // 贡献的高斯点数量
	const float* __restrict__ bg_color,          // 背景颜色
	float* __restrict__ out_color,               // 输出颜色
	float* __restrict__ out_feature,         // //额外特征输出
	float* __restrict__ out_others               // 其他辅助输出
)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();      // 获取当前线程块
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // 水平方向上的线程块数量
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // 当前瓦片的最小像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) }; // 当前瓦片的最大像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // 当前线程处理的像素坐标
	uint32_t pix_id = W * pix.y + pix.x; // 当前线程处理的像素ID
	float2 pixf = { (float)pix.x, (float)pix.y }; // 当前线程处理的像素坐标（浮点型）
	const float2 ray = { (pixf.x - W * 0.5) / focal_x, (pixf.y - H * 0.5) / focal_x };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H; // 判断当前线程是否处理有效像素
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside; // 无效像素的线程标记为done，用于帮助数据获取但不进行光栅化

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x]; // 获取当前瓦片的点范围
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); // 计算需要处理的批次数
	int toDo = range.y - range.x; // 当前瓦片中需要处理的点数量

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE]; // 存储批次中的点ID
	__shared__ float2 collected_xy[BLOCK_SIZE]; // 存储批次中的点位置
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE]; // 存储批次中的点法线和不透明度
	__shared__ float3 collected_Tu[BLOCK_SIZE]; // 存储批次中的点变换矩阵Tu
	__shared__ float3 collected_Tv[BLOCK_SIZE]; // 存储批次中的点变换矩阵Tv
	__shared__ float3 collected_Tw[BLOCK_SIZE]; // 存储批次中的点变换矩阵Tw

	// Initialize helper variables
	float T = 1.0f; // 初始化透明度
	uint32_t contributor = 0; // 初始化贡献者数量
	uint32_t last_contributor = 0; // 初始化最后贡献者数量
	float C[CHANNELS] = { 0 }; // 初始化颜色
	float F[MAX_FEATURES] = { 0 }; // // 初始化额外特征 

#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0}; // 初始化法线
	float D = { 0 }; // 初始化深度
	float M1 = {0}; // 初始化M1
	float M2 = {0}; // 初始化M2
	float distortion = {0}; // 初始化失真
	float median_depth = {0}; // 初始化中值深度
	float median_contributor = {-1}; // 初始化中值贡献者数量
#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done); // 统计done线程数量
		if (num_done == BLOCK_SIZE)
			break; // 如果所有线程都done，则退出循环

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank(); // 计算当前线程的进度
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress]; // 获取当前点ID
			collected_id[block.thread_rank()] = coll_id; // 存储当前点ID
			collected_xy[block.thread_rank()] = points_xy_image[coll_id]; // 存储当前点位置
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id]; // 存储当前点法线和不透明度
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]}; // 存储当前点变换矩阵Tu
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]}; // 存储当前点变换矩阵Tv
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]}; // 存储当前点变换矩阵Tw
		}
		block.sync(); // 同步线程块中的所有线程

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++; // 记录当前贡献者数量

			// First compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j]; // 获取当前点位置
			const float3 Tu = collected_Tu[j]; // 获取当前点变换矩阵Tu
			const float3 Tv = collected_Tv[j]; // 获取当前点变换矩阵Tv
			const float3 Tw = collected_Tw[j]; // 获取当前点变换矩阵Tw
			float3 k = pix.x * Tw - Tu; // 计算k向量
			float3 l = pix.y * Tw - Tv; // 计算l向量
			float3 p = cross(k, l); // 计算p向量
			if (p.z == 0.0) continue; // 如果p向量的z分量为0，则跳过
			float2 s = {p.x / p.z, p.y / p.z}; // 计算s向量
			float rho3d = (s.x * s.x + s.y * s.y); // 计算rho3d
			float2 d = {xy.x - pixf.x, xy.y - pixf.y}; // 计算d向量
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); // 计算rho2d

			// compute intersection and depth
			float rho = min(rho3d, rho2d); // 计算rho
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; // 计算深度
			if (depth < near_n) continue; // 如果深度小于近剪裁面，则跳过
			float4 nor_o = collected_normal_opacity[j]; // 获取法线和不透明度
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z}; // 获取法线
			float opa = nor_o.w; // 获取不透明度

			float power = -0.5f * rho; // 计算power
			if (power > 0.0f)
				continue; // 如果power大于0，则跳过

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opa * exp(power)); // 计算alpha
			if (alpha < 1.0f / 255.0f)
				continue; // 如果alpha小于1/255，则跳过
			float test_T = T * (1 - alpha); // 计算test_T
			if (test_T < 0.0001f)
			{
				done = true; // 如果test_T小于0.0001，则标记为done
				continue;
			}

			float w = alpha * T; // 计算权重w
#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1 - T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch = 0; ch < 3; ch++) N[ch] += normal[ch] * w;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += colors[collected_id[j] * CHANNELS + ch] * w; // 计算颜色
			// // 计算额外特征
			for (int ch = 0; ch < S; ch++)
				F[ch] += features[collected_id[j] * S + ch] * w;


			T = test_T; // 更新透明度T

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor; // 更新最后贡献者数量
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T; // 存储最终透明度
		n_contrib[pix_id] = last_contributor; // 存储最后贡献者数量
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // 存储颜色
		for (int ch = 0; ch < S; ch++)
			out_feature[ch * H * W + pix_id] = F[ch];   // // 存储额外特征

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor; // 存储中值贡献者数量
		final_T[pix_id + H * W] = M1; // 存储M1
		final_T[pix_id + 2 * H * W] = M2; // 存储M2
		out_others[pix_id + DEPTH_OFFSET * H * W] = D; // 存储深度
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T; // 存储alpha
		for (int ch = 0; ch < 3; ch++) out_others[pix_id + (NORMAL_OFFSET + ch) * H * W] = N[ch]; // 存储法线
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth; // 存储中值深度
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion; // 存储失真
		out_others[pix_id + UNBIASED_DEPTH_OFFSET * H * W] = F[S-1] / -(N[0] * ray.x + N[1] * ray.y + N[2] + 1.0e-8);;
#endif
	}
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const int S, int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* features,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_feature,
	float* out_others)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		S, W, H,
		focal_x, focal_y,
		means2D,
		colors,
		features,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_feature,
		out_others);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
