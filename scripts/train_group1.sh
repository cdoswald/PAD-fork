CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name cheetah \
    --task_name run \
    --action_repeat 4 \
    --mode train \
    --train_steps 200000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/cheetah_run/inv/seed0 \
    --save_model

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name hopper \
    --task_name hop \
    --action_repeat 4 \
    --mode train \
    --train_steps 200000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/hopper_hop/inv/seed0 \
    --save_model

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name walker \
    --task_name walk \
    --action_repeat 4 \
    --mode train \
    --train_steps 200000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/walker_walk/inv/seed0 \
    --save_model

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name walker \
    --task_name run \
    --action_repeat 4 \
    --mode train \
    --train_steps 200000 \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/walker_run/inv/seed0 \
    --save_model
