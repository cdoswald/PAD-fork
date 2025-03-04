##################################
# Inverse dynamics auxiliary task
##################################
for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
	--domain_name cartpole \
	--task_name swingup \
	--action_repeat 8 \
	--mode color_easy \
	--use_inv \
	--ss_update_quantities "1,2,4,8,16" \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cartpole_swingup/inv/0_freeze_decoder \
	--pad_checkpoint 500k \
	--pad_num_episodes 100 \
	--pad_reset_agent "episode" \
	--freeze_aux_decoder
done

for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
	--domain_name cartpole \
	--task_name swingup \
	--action_repeat 8 \
	--mode color_easy \
	--use_inv \
	--ss_update_quantities "1,2,4,8,16" \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cartpole_swingup/inv/0_freeze_decoder \
	--pad_checkpoint 500k \
	--pad_num_episodes 100 \
	--pad_reset_agent "none" \
	--freeze_aux_decoder
done

for seed in {1..10} 
do
    CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
	--domain_name cartpole \
	--task_name swingup \
	--action_repeat 8 \
	--mode color_easy \
	--use_inv \
	--ss_update_quantities "1,2,4,8,16" \
	--num_shared_layers 8 \
	--seed $seed \
	--work_dir logs/cartpole_swingup/inv/0_freeze_decoder \
	--pad_checkpoint 500k \
	--pad_num_episodes 100 \
	--pad_reset_agent "ss_updates" \
	--freeze_aux_decoder
done
