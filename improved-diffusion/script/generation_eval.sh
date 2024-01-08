export CUDA_VISIBLE_DEVICES=1
python "scripts/infill_eval_coco.py" \
    --model_path diffusion_model/coco/1125/ema_0.9999_200000.pt \
    --eval_task 'control_pos' \
    --use_ddim True\
    --notes "tree_adagrad" \
    --eta 1. \
    --verbose pipe \

