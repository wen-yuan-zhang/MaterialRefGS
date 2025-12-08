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
    scene=gardenspheres # change this to your scene name
    data_dir=/data14_2/tjm/code/ref-gaussian/data/refreal
    output_dir=output_refreal
    
    # Prior paths
    metric3d_path=/data14_2/tjm/code/Metric3D/output
    ref_score_path=/data14_2/tjm/code/ref-gaussian-tjm/tmp_ref_scores_merge_mask2
    
    python train_refreal.py \
        -s ${data_dir}/${scene}\
        --iterations 30000 \
        --indirect_from_iter 12500 \
        --volume_render_until_iter 0 \
        --initial 1 \
        --init_until_iter 3000 \
        -r 4 \
        --normal_loss_start 7000\
        --densify_until_iter 20000\
        --normal_prop_until_iter 18000\
        --lambda_normal_smooth 0\
        --ncc_scale 0.5\
        --eval\
        --lambda_normal_render_depth 0.05\
        --multi_view_weight_from_iter 7000\
        --multi_view_ncc_weight 0.15\
        --lambda_dist 1000\
        --perceptual_loss_start_iter 16000\
        -m ${output_dir}/${scene}\
        --ref_score_loss_weight 0.01\
        --metric3d_path ${metric3d_path}\
        --ref_score_path ${ref_score_path}
fi