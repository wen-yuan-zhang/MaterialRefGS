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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int S, int W, int H,   // //
		float focal_x, float focal_y,
		const float* bg_color,
		const float2* means2D,
		const float4* normal_opacity,
		const float* colors,
		const float* features, // //
		const float* transMats,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dfeatures, // //
		const float* dL_depths,
		float * dL_dtransMat,
		float3* dL_dmean2D,
		float* dL_dnormal3D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_dfeature // //
		);

	void preprocess(
		int P, int D, int M,
		const float3* means3D,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec2* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* transMats,
		const float* viewmatrix,
		const float* projmatrix,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		const glm::vec3* campos, 
		float3* dL_dmean2Ds,
		const float* dL_dnormal3Ds,
		float* dL_dtransMats,
		float* dL_dcolors,
		float* dL_dshs,
		glm::vec3* dL_dmean3Ds,
		glm::vec2* dL_dscales,
		glm::vec4* dL_drots);
}

#endif
