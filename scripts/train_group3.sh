CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name acrobot \
    --task_name swingup \
    --action_repeat 4 \
    --mode train \
    --train_steps 200000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/acrobot_swingup/inv/seed0 \
    --save_model

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name ball_in_cup \
    --task_name catch \
    --action_repeat 4 \
    --mode train \
    --train_steps 100000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/ball_in_cup_catch/inv/seed0 \
    --save_model
