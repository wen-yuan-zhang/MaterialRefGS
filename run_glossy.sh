#!/bin/bash

# Glossy Synthetic Training Script
# This script trains on the Glossy Synthetic dataset

gpuid=7
if [ "$gpuid" -ge 0 ]; then
    export CUDA_VISIBLE_DEVICES=${gpuid}
    
    # Scene configuration
    scene=tbell  # Change this to your scene name
    data_dir=/data14_2/tjm/code/ref-gaussian/data/GlossySynthetic_blender
    
    # Prior paths
    geowizard_path=/data14_2/tjm/code/GeoWizard/output
    metric3d_path=/data14_2/tjm/code/Metric3D/output_glossy
    idarb_path=/data14_2/tjm/code/IDArb/output
    
    # Output directory
    output_dir=./output_glossy
    
    OMP_NUM_THREADS=4 python train_glossy.py \
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
        --lambda_perceptual_loss 0.05 \
        --geowizard_path ${geowizard_path} \
        --metric3d_path ${metric3d_path} \
        --idarb_path ${idarb_path}
fi