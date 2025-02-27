# Cartpole Swingup - Inverse Dynamics
for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval_train_env.py \
	--domain_name cartpole \
	--task_name swingup \
	--action_repeat 8 \
	--mode train \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cartpole_swingup/inv/0 \
	--pad_checkpoint 500k \
	--pad_num_episodes 100 \
	--pad_reset_agent "episode"
done

# Cartpole Swingup - Rotation
for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval_train_env.py \
	--domain_name cartpole \
	--task_name swingup \
	--action_repeat 8 \
	--mode train \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cartpole_swingup/rot/0 \
	--pad_checkpoint 100k \
	--pad_num_episodes 100 \
	--pad_reset_agent "episode"
done

# Cheetah Run - Inverse Dynamics
for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval_train_env.py \
	--domain_name cheetah \
	--task_name run \
	--action_repeat 4 \
	--mode train \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cheetah_run/inv/0 \
	--pad_checkpoint 100k \
	--pad_num_episodes 100 \
	--pad_reset_agent "episode"
done

# Cheetah Run - Rotation
for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval_train_env.py \
	--domain_name cheetah \
	--task_name run \
	--action_repeat 4 \
	--mode train \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cheetah_run/rot/0 \
	--pad_checkpoint 100k \
	--pad_num_episodes 100 \
	--pad_reset_agent "episode"
done
