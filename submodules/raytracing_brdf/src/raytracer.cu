#pragma once

#include <raytracing/raytracer.h>

#include <raytracing/common.h>
#include <raytracing/bvh.cuh>

#include <Eigen/Dense>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <omp.h>  

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace raytracing {

class RayTracerImpl : public RayTracer {
public:

    // accept numpy array (cpu) to init 
    RayTracerImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles) : RayTracer() {

        const size_t n_vertices = vertices.rows();
        const size_t n_triangles = triangles.rows();

        triangles_cpu.resize(n_triangles);
        triangles_cpu_bak.resize(n_triangles);
        
        omp_set_num_threads(44);

        float startTime = omp_get_wtime();

        // #pragma omp parallel for num_threads(44)
        for (size_t i = 0; i < n_triangles; i++) {
            // triangles_cpu[i] = {vertices.row(triangles(i, 0)), vertices.row(triangles(i, 1)), vertices.row(triangles(i, 2))};
            // triangles_cpu_bak[i] = {{vertices.row(triangles(i, 0)), vertices.row(triangles(i, 1)), vertices.row(triangles(i, 2))},i};
            auto v0 = vertices.row(triangles(i, 0));  
            auto v1 = vertices.row(triangles(i, 1));  
            auto v2 = vertices.row(triangles(i, 2));  
            triangles_cpu[i] = {v0, v1, v2};
            triangles_cpu_bak[i] = {{v0, v1, v2}, i};  
        }
        // float endTime = omp_get_wtime();
        // std::cout<<"Initial Time"<<endTime - startTime<<std::endl; 
        // std::cout.flush();

        // startTime = omp_get_wtime();
        if (!triangle_bvh) {
            triangle_bvh = TriangleBvh::make();
        }
        // endTime = omp_get_wtime();
        // std::cout<<"TriangleBVh Build"<<endTime - startTime<<std::endl; 
        // std::cout.flush();

        // std::cout<<"Finish Triangle BVH Initialization"<<std::endl;

        triangle_bvh->build(triangles_cpu, 8);

        create_faceids_mappings();

        triangles_gpu.resize_and_copy_from_host(triangles_cpu);

        // TODO: need OPTIX
        // triangle_bvh->build_optix(triangles_gpu, m_inference_stream);
        // std::cout.flush();
    }

    void create_faceids_mappings() {
        const int N = triangles_cpu.size();
        std::vector<std::pair<Triangle,unsigned>> triangles_cpu_bvh_bak(N,{{}, 0});
        triangels_ids_mapping.resize(N);
        
        #pragma omp parallel for num_threads(44)
        for (size_t i=0;i < N;i++) {
            triangles_cpu_bvh_bak[i] = {triangles_cpu[i],i};
        }
        auto compare = [](const std::pair<Triangle, unsigned>& a, const std::pair<Triangle, unsigned>& b) {
            if (a.first < b.first) {
                return true;
            } else if (b.first < a.first) {
                return false;
            } else {
                return a.second < b.second;
            }
        };

        std::sort(triangles_cpu_bvh_bak.begin(),triangles_cpu_bvh_bak.end(),compare);
        std::sort(triangles_cpu_bak.begin(),triangles_cpu_bak.end(),compare);

        std::cout<<"valid untile here"<<std::endl;
        std::cout<<triangles_cpu_bvh_bak.size()<<" "<<triangles_cpu_bak.size()<<std::endl;
        std::cout.flush();

        #pragma omp parallel for num_threads(44)
        for (size_t i=0;i < N;i++) {
            assert((triangles_cpu_bvh_bak[i].second<N) and (triangles_cpu_bak[i].second<N));
            assert((triangles_cpu_bvh_bak[i].second>=0) and (triangles_cpu_bvh_bak[i].second>=0));
            triangels_ids_mapping[triangles_cpu_bvh_bak[i].second] = triangles_cpu_bak[i].second;
        }
        std::cout<<"Finish Building FaceIds Mapping"<<std::endl;
        std::cout.flush();
    }

    // accept torch tensor (gpu) to init
    void trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor normals, at::Tensor depth, at::Tensor triangle_indices) {

        // must be contiguous, float, cuda, shape [N, 3]. check in torch side.

        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->ray_trace_gpu(n_elements, rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), positions.data_ptr<float>(), normals.data_ptr<float>(), depth.data_ptr<float>(), triangle_indices.data_ptr<int>(), triangles_gpu.data(), stream);

        cudaStreamSynchronize(stream);

        // std::cout << "Size of triangle_indices: " << triangle_indices.size(0) << std::endl;  

        // std::cout.flush();
        int * triangle_indices_ptr = triangle_indices.data_ptr<int>();
        size_t l =  triangle_indices.size(0);
        // std::cout<<l<<" "<<n_elements<<std::endl;

        // #pragma omp parallel for num_threads(44)
        // int count = 0;
        // for (size_t i=0;i<n_elements;i++) {
        //     // assert((triangle_indices_ptr[i]<triangels_ids_mapping.size()) and (triangle_indices_ptr[i] >= 0));
        //     ++count;
        //     const int v = triangle_indices_ptr[i];
        //     std::cout<<v<<std::endl;
        //     // if (v<-1||v>=triangels_ids_mapping.size()) {
        //     //     std::cout<<v<<std::endl;
        //     //     throw std::invalid_argument("Invalid value in triangle_indices: " + std::to_string(v));  
        //     // }
        //     // if (v<0) continue;
        //     // triangle_indices_ptr[i] = triangels_ids_mapping[v];
        // }
    }

    at::Tensor get_triangels_ids_mapping() {
        const int N = triangels_ids_mapping.size();
        std::vector<int64_t> shape = {N};
        at::Tensor mappings = torch::empty(shape, torch::kInt); 
        int * data_ptr = mappings.data_ptr<int>();
        
        #pragma omp parallel for num_threads(44)
        for (size_t i=0;i<N;i++) {
            data_ptr[i] = triangels_ids_mapping[i];
        }
        return mappings;
    }

    at::Tensor get_triangles() {
        const int N = triangles_cpu.size(); // (N,3,3)
        triangles_gpu.copy_to_host(triangles_cpu,N);
        std::vector<int64_t> shape = {N, 3, 3};  
        at::Tensor triangles_tensor = torch::zeros(shape, torch::kFloat); 
        for (int i=0;i<N;i++) {
            // std::cout<<"I: "<<i<<std::endl;
            const Triangle tr = triangles_cpu[i]; // (3,3)
            for (int j=0;j<3;j++) {
                // triangles_tensor[i,0,j] = tr.a[j];
                triangles_tensor.index_put_({i,0,j},tr.a[j]);
                // std::cout<<tr.a[j]<<" "<<triangles_tensor[i,0,j]<<" ";
            }
            // std::cout<<std::endl;
            for (int j=0;j<3;j++) {
                // triangles_tensor[i,1,j] = tr.b[j];
                triangles_tensor.index_put_({i,1,j},tr.b[j]);
                // std::cout<<tr.b[j]<<" "<<triangles_tensor[i,1,j]<<" ";
            }
            // std::cout<<std::endl;
            for (int j=0;j<3;j++) {
                // triangles_tensor[i,2,j] = tr.c[j];
                triangles_tensor.index_put_({i,2,j},tr.c[j]);
                // std::cout<<tr.b[j]<<" "<<triangles_tensor[i,2,j]<<" ";
            }
            // std::cout<<std::endl;
        }
        // for (int i=0;i<N;i++) {
        //     for (int j=0;j<3;j++) {
        //         for (int k=0;k<3;k++) {
        //             std::cout<<triangles_tensor[i,j,k]<<" ";
        //         }
        //         std::cout<<std::endl;
        //     }
        // }
        // std::cout.flush();
        return triangles_tensor; 
    }

    std::vector<Triangle> triangles_cpu;
    std::vector<std::pair<Triangle,unsigned>> triangles_cpu_bak;
    std::vector<unsigned long long> triangels_ids_mapping; //
    GPUMemory<Triangle> triangles_gpu;
    std::shared_ptr<TriangleBvh> triangle_bvh;
};
    
RayTracer* create_raytracer(Ref<const Verts> vertices, Ref<const Trigs> triangles) {
    return new RayTracerImpl{vertices, triangles};
}

} // namespace raytracing