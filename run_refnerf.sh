#!/bin/bash

# RefNeRF Training Script
# This script trains the RefNeRF model with specified parameters

gpuid=0
if [ "$gpuid" -ge 0 ]; then
    export CUDA_VISIBLE_DEVICES=${gpuid}
    
    # Scene configuration
    scene=toaster  # Change this to your scene name (e.g., ball, car, coffee, helmet, teapot, toaster)
    data_dir=/data12_1/tjm/code/matrefgs_release_v2/data2/refnerf
    
    # Prior paths
    geowizard_path=/data14_2/tjm/code/GeoWizard/output
    metric3d_path=/data14_2/tjm/code/Metric3D/output_refnerf
    idarb_path=/data14_2/tjm/code/IDArb/output
    
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
        --geowizard_path ${geowizard_path} \
        --metric3d_path ${metric3d_path} \
        --idarb_path ${idarb_path}
fi