from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch

from arguments import parse_args


if __name__ == "__main__":

	args = parse_args()

	# Visualize average episode rewards
	work_dir = "logs/cartpole_swingup/inv/0"
	mode = "color_easy"
	results_fp_pattern = os.path.join(work_dir, f'pad_{mode}_evalseed_*.pt')
	seed_results_files = glob(results_fp_pattern)

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

		# Update plotting parameters
		max_y = max(max_y, np.max(rewards_episode_by_avg_seed))

		# Plot results
		sns.lineplot(
			x=range(len(rewards_episode_by_avg_seed)),
			y=rewards_episode_by_avg_seed,
			label=plot_update_step,
			ax=axes
		)
	axes.legend(
		title="Aux Updates",
		loc="center left",
		bbox_to_anchor=(1,0.5),
	)
	axes.set_ylim(0, int(max_y * 1.1))
	axes.set_xlabel("Episode")
	axes.set_ylabel("Average Reward")
	axes.set_title("")
