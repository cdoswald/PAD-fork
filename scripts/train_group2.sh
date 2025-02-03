CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name quadruped \
    --task_name walk \
    --action_repeat 4 \
    --mode train \
    --train_steps 200000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/quadruped_walk/inv/seed0 \
    --save_model

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name quadruped \
    --task_name run \
    --action_repeat 4 \
    --mode train \
    --train_steps 200000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/quadruped_run/inv/seed0 \
    --save_model

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name humanoid \
    --task_name stand \
    --action_repeat 4 \
    --mode train \
    --train_steps 200000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/humanoid_stand/inv/seed0 \
    --save_model

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name humanoid \
    --task_name walk \
    --action_repeat 4 \
    --mode train \
    --train_steps 200000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/humanoid_walk/inv/seed0 \
    --save_model
