##################################
# Rotation auxiliary task
##################################
for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
	--domain_name cartpole \
	--task_name swingup \
	--action_repeat 8 \
	--mode color_hard \
	--use_rot \
	--ss_update_quantities "1,2,4,8,16" \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cartpole_swingup/rot/0 \
	--pad_checkpoint 100k \
	--pad_num_episodes 100 \
	--pad_reset_agent "episode"
done

for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
	--domain_name cartpole \
	--task_name swingup \
	--action_repeat 8 \
	--mode color_hard \
	--use_rot \
	--ss_update_quantities "1,2,4,8,16" \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cartpole_swingup/rot/0 \
	--pad_checkpoint 100k \
	--pad_num_episodes 100 \
	--pad_reset_agent "none"
done

for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
	--domain_name cartpole \
	--task_name swingup \
	--action_repeat 8 \
	--mode color_hard \
	--use_rot \
	--ss_update_quantities "1,2,4,8,16" \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cartpole_swingup/rot/0 \
	--pad_checkpoint 100k \
	--pad_num_episodes 100 \
	--pad_reset_agent "ss_updates"
done
