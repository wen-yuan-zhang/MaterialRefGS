#!/bin/bash

# RefNeRF Training Script
# This script trains the RefNeRF model with specified parameters

gpuid=-1
for ((i=1; i<=9; i++)); do
    stat2=$(gpustat | awk '{print $11}' | sed -n "${i}p" 2>/dev/null)
    if [ -n "$stat2" ] && [[ "$stat2" =~ ^[0-9]+$ ]] && [ "$stat2" -lt 100 ]; then
        echo "running on gpu $((i-2))"
        gpuid=$((i-2))
        break
    fi
done

if [ "$gpuid" -ge 0 ]; then
    export CUDA_VISIBLE_DEVICES=${gpuid}
    
    # Scene configuration
    scene=toaster  # Change this to your scene name (e.g., ball, car, coffee, helmet, teapot, toaster)
    data_dir=/data12_1/tjm/code/matrefgs_release_v2/data2/refnerf
    
    # Prior paths
    metric3d_path=/data14_2/tjm/code/Metric3D/output_refnerf
    
    # Output directory
    output_dir=./output_refnerf
    
    OMP_NUM_THREADS=4 python train_refnerf.py \
        -s ${data_dir}/${scene} \
        -m ${output_dir}/${scene} \
        --quiet \
        --iterations 50000 \
        --indirect_from_iter 20000 \
        --volume_render_until_iter 0 \
        --initial 1 \
        --init_until_iter 3000 \
        --normal_loss_start 3000 \
        --normal_prop_until_iter 30000 \
        --densify_until_iter 30000 \
        --lambda_normal_smooth 0 \
        --ncc_scale 1.0 \
        --eval \
        --ref_score_start_iter 50000 \
        --white_background \
        --no-use_perceptual_loss \
        --no-use_metallic_warp_loss \
        --no-use_roughness_warp_loss \
        --metric3d_path ${metric3d_path} \
fi