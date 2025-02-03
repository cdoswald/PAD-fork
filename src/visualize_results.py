from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch

from arguments import parse_args


if __name__ == "__main__":

	args = parse_args()

	work_dir = "logs/cartpole_swingup/inv/0"
	mode = "color_easy"
	results_fp_pattern = os.path.join(work_dir, f'pad_{mode}_evalseed_*.pt')
	# results_fp_pattern = os.path.join(args.work_dir, f'pad_{args.mode}_evalseed_*.pt')

	seed_results_files = glob(results_fp_pattern)

	# Visualize average episode rewards
	plot_CI = True
	plot_stddev = False
	max_y = 0
	fig, axes = plt.subplots(1, 1, figsize=(8,4))
	plot_update_steps = [0, 1, 2, 4, 8, 16]
	for plot_update_step in plot_update_steps:

		# Get results for all seeds for plot_update_step value
		temp_list = []
		for seed_i_result_file in seed_results_files:
			seed_i_results = torch.load(seed_i_result_file, weights_only=False)
			if plot_update_step == 0:
				temp_list.append(seed_i_results["eval_episode_rewards"])
			else:
				temp_list.append(seed_i_results["pad_episode_rewards"][plot_update_step])

		# Average across seeds
		rewards_episode_by_seed = np.array(temp_list).T
		rewards_episode_by_avg_seed = np.mean(rewards_episode_by_seed, axis=1)
		
		# Compute standard deviation
		rewards_stddev_avg_seed = np.std(rewards_episode_by_seed, axis=1)

		# Compute confidence interval
		n_seeds = rewards_episode_by_seed.shape[0]
		lower_bound_seed_idx = round(n_seeds * 0.1)
		upper_bound_seed_idx = round(n_seeds * 0.9)
		rewards_episode_by_seed_sorted = np.sort(rewards_episode_by_seed, axis=1)
		rewards_episode_lower_bound = rewards_episode_by_seed_sorted[:, lower_bound_seed_idx]
		rewards_episode_upper_bound = rewards_episode_by_seed_sorted[:, upper_bound_seed_idx]
		
		# Update plotting parameters
		n_episodes = len(rewards_episode_by_avg_seed)
		max_y = max(max_y, np.max(rewards_episode_by_avg_seed))

		# Plot results
		sns.lineplot(
			x=range(n_episodes),
			y=rewards_episode_by_avg_seed,
			label=plot_update_step,
			ax=axes
		)
		if plot_CI:
			axes.fill_between(
				x=range(n_episodes),
				y1=rewards_episode_lower_bound,
				y2=rewards_episode_upper_bound,
				alpha=0.2
			)
		if plot_stddev:
			axes.fill_between(
				x=range(n_episodes),
				y1=rewards_episode_by_avg_seed - rewards_stddev_avg_seed,
				y2=rewards_episode_by_avg_seed + rewards_stddev_avg_seed,
				alpha=0.2,
				linestyle="dashed"
			)
	
	# Format plot
	axes.legend(
		title="Aux Updates",
		loc="center left",
		bbox_to_anchor=(1,0.5),
	)
	axes.set_ylim(0, int(max_y * 1.1))
	axes.set_xlabel("Episode")
	axes.set_ylabel("Average Reward")
	axes.set_title("")
