from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import torch



if __name__ == "__main__":

	color_dict = {
		0:"tab:blue",
		1:"tab:orange",
		2:"tab:green",
		4:"tab:red",
		8:"tab:purple",
		16:"tab:brown"
	}

	domain = "cartpole"
	task = "swingup"
	action_repeats = 8
	steps_per_episode = 1000
	work_dir = "logs/cartpole_swingup/inv/0"
	modes = ["color_easy", "color_hard"]
	reset_agent_modes = ["episode", "none"]
	plot_update_steps = [0, 1, 2, 4, 8, 16]

	plot_CI = False
	plot_stddev = False
	include_title = False

	gaussian_filter_stddev = 1

	for gaussian_filter in [True, False]:
		for mode in modes:
			sub_dirs = [f"reset_{mode}" for mode in reset_agent_modes]
			for sub_dir in sub_dirs:
				results_dir = os.path.join(work_dir, sub_dir)
				results_fp_pattern = os.path.join(results_dir, f'pad_{mode}_evalseed_*.pt')
				seed_results_files = glob(results_fp_pattern)

				if seed_results_files:

					# Visualize average episode rewards
					rewards_min_y = float("inf")
					rewards_max_y = float("-inf")
					fig, axes = plt.subplots(2, 1, figsize=(8,6))
					fig.subplots_adjust(hspace=0.3)
					for plot_update_step in plot_update_steps:

						# Get results for all seeds for plot_update_step value
						temp_reward_list = []
						temp_ssl_loss_list = []
						for seed_i_result_file in seed_results_files:
							seed_i_results = torch.load(seed_i_result_file, weights_only=False)
							if plot_update_step == 0:
								temp_reward_list.append(seed_i_results["eval_episode_rewards"])
							else:
								temp_reward_list.append(
									seed_i_results["pad_episode_rewards"][plot_update_step]
								)
								temp_ssl_loss_list.append(
									seed_i_results["pad_ssl_losses"][plot_update_step]
								)

						# Calculate average rewards across seeds
						rewards_episode_by_seed = np.array(temp_reward_list).T
						rewards_episode_by_avg_seed = np.mean(rewards_episode_by_seed, axis=1)
						
						# Calculate standard deviation of rewards across seeds
						rewards_stddev_avg_seed = np.std(rewards_episode_by_seed, axis=1)

						# Calculate confidence interval
						n_seeds = rewards_episode_by_seed.shape[1]
						lower_bound_seed_idx = round(n_seeds * 0.1)
						upper_bound_seed_idx = round(n_seeds * 0.9)
						rewards_episode_by_seed_sorted = np.sort(rewards_episode_by_seed, axis=1)
						rewards_episode_lower_bound = rewards_episode_by_seed_sorted[:, lower_bound_seed_idx]
						rewards_episode_upper_bound = rewards_episode_by_seed_sorted[:, upper_bound_seed_idx]

						# Calculate average auxiliary task loss across seeds and steps
						if plot_update_step != 0:
							aux_loss_episode_by_update_by_seed = (
								np.array(temp_ssl_loss_list).transpose(1, 2, 0)
							)
							aux_loss_episode_by_update_by_avg_seed = np.mean(
								aux_loss_episode_by_update_by_seed, axis=-1,
							)
							aux_loss_episode_by_avg_update_by_avg_seed = np.mean(
								aux_loss_episode_by_update_by_avg_seed, axis=-1
							)

							# # Expand aux loss array to account for action repeats
							# aux_loss_episode_by_update_by_avg_seed_expanded = np.repeat(
							# 	aux_loss_episode_by_update_by_avg_seed,
							# 	repeats=action_repeats,
							# 	axis=1,
							# )
							# total_updates_per_episode = (
							# 	aux_loss_episode_by_update_by_avg_seed_expanded.shape[-1]
							# )
							# assert (total_updates_per_episode / steps_per_episode == plot_update_step)

						# Update plotting parameters
						n_episodes = len(rewards_episode_by_avg_seed)
						rewards_min_y = min(rewards_min_y, np.min(rewards_episode_by_avg_seed))
						rewards_max_y = max(rewards_max_y, np.max(rewards_episode_by_avg_seed))
						plot_color = color_dict[plot_update_step]

						# Smooth values (if applicable)
						if gaussian_filter:
							rewards_episode_by_avg_seed = gaussian_filter1d(
								rewards_episode_by_avg_seed,
								gaussian_filter_stddev,
							)
							if plot_update_step != 0:
								aux_loss_episode_by_avg_update_by_avg_seed = gaussian_filter1d(
									aux_loss_episode_by_avg_update_by_avg_seed,
									gaussian_filter_stddev,
								)

						# Plot rewards
						sns.lineplot(
							x=range(n_episodes),
							y=rewards_episode_by_avg_seed,
							label=plot_update_step,
							ax=axes[0],
							color=plot_color,
							legend=False,
						)
						if plot_CI:
							axes[0].fill_between(
								x=range(n_episodes),
								y1=rewards_episode_lower_bound,
								y2=rewards_episode_upper_bound,
								alpha=0.2,
								color=plot_color,
							)
						if plot_stddev:
							axes[0].fill_between(
								x=range(n_episodes),
								y1=rewards_episode_by_avg_seed - rewards_stddev_avg_seed,
								y2=rewards_episode_by_avg_seed + rewards_stddev_avg_seed,
								alpha=0.2,
								linestyle="dashed",
								color=plot_color,
							)
						
						# Plot auxiliary losses
						if plot_update_step != 0:
							sns.lineplot(
								x=range(n_episodes),
								y=aux_loss_episode_by_avg_update_by_avg_seed,
								label=plot_update_step,
								ax=axes[1],
								legend=False,
								color=plot_color,
							)

					# Format plot
					axes[0].set_ylim(0, int(rewards_max_y * 1.05))
					axes[0].set_ylabel("Avg Reward")
					axes[1].set_ylabel("Avg Auxiliary Loss")
					for i in range(len(axes)):
						axes[i].set_xlabel("Episode")
					# Create legend
					handles, labels = fig.axes[0].get_legend_handles_labels()
					fig.legend(
						handles,
						labels,
						title="Aux Updates",
						loc="center left",
						bbox_to_anchor=(0.92,0.5),
					)
					if include_title:
						axes.set_title(
							f"Average Episode Reward for {domain.title()} {task.title()}" +
							f"\nMode: {mode}; # seeds: {n_seeds}"
						)
					suffix = "_smoothed" if gaussian_filter else ""
					fig_name = f"avg_episode_reward_{domain.title()}_{task.title()}_mode_{mode}_{n_seeds}_seeds{suffix}.png"
					save_path = os.path.join(results_dir, fig_name)
					fig.savefig(save_path, bbox_inches="tight")