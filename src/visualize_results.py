from glob import glob
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import torch


if __name__ == "__main__":

	plot_CI = False
	plot_stddev = False
	include_title = False

	steps_per_episode = 1000
	reset_agent_modes = ["episode", "none", "ss_updates"]
	plot_update_steps = [0, 1, 2, 4, 8, 16]

	rewards_gaussian_filter_stddev = 1
	aux_loss_gaussian_filter_stddev = 1
	enc_dist_gaussian_filter_stddev = 100

	color_dict = {
		0:"tab:blue",
		1:"tab:orange",
		2:"tab:green",
		4:"tab:red",
		8:"tab:purple",
		16:"tab:brown"
	}

	domain_task_list = [
		{
			"domain":"cartpole",
			"task":"swingup",
			"aux_models": ["inv", "rot"],
			"color_modes":["color_easy"],
		},
		{
			"domain":"cheetah",
			"task":"run",
			"aux_models": ["inv", "rot"],
			"color_modes":["color_easy"],
		},
	]

	start_time = time.strftime('%Y-%m-%d %H:%M:%S')
	for params in domain_task_list:
		domain = params["domain"]
		task = params["task"]
		for aux_model in params["aux_models"]:
			work_dir = f"logs/{domain}_{task}/{aux_model}/0"
			sub_dirs = [f"reset_{mode}" for mode in reset_agent_modes]
			for sub_dir in sub_dirs:
				for color_mode in params["color_modes"]:
					results_dir = os.path.join(work_dir, sub_dir)
					results_fp_pattern = os.path.join(results_dir, f'pad_{color_mode}_evalseed_*.pt')
					seed_results_files = glob(results_fp_pattern)

					if seed_results_files:

						# Load data
						data_dict = {}
						for plot_update_step in plot_update_steps:
							print(
								f"Loading data for {domain} {task}, {aux_model}, "+
								f"{color_mode}, {sub_dir}, {plot_update_step} aux updates"
							)
							# Get results for all seeds for plot_update_step value
							temp_reward_list = []
							temp_ssl_loss_list = []
							temp_actor_embed_dists_list = []
							temp_shared_embed_dists_list = []
							for seed_i_result_file in seed_results_files:
								seed_i_results = torch.load(seed_i_result_file, weights_only=False)
								# Get episode rewards
								temp_reward_list.append(seed_i_results["episode_rewards"][plot_update_step])
								# Get auxiliary losses
								if plot_update_step != 0:
									temp_ssl_loss_list.append(
										seed_i_results["aux_losses"][plot_update_step]
									)
								# Get encoder distances
								temp_actor_embed_dists_list.append([
									x.item() if isinstance(x, torch.Tensor) else x for x in 
									seed_i_results["embed_dists"][plot_update_step]["actor_enc"]
								])
								temp_shared_embed_dists_list.append([
									x.item() if isinstance(x, torch.Tensor) else x for x in 
									seed_i_results["embed_dists"][plot_update_step]["shared_conv_enc"]
								])

							# Concatenate across seeds
							temp_rewards_episode_by_seed = np.array(temp_reward_list).T
							temp_actor_enc_dists_step_by_seed = np.array(temp_actor_embed_dists_list).T
							temp_shared_enc_dists_step_by_seed = np.array(temp_shared_embed_dists_list).T
							if plot_update_step != 0:
								temp_aux_loss_episode_by_update_by_seed = (
									np.array(temp_ssl_loss_list).transpose(1, 2, 0)
								)
							else:
								temp_aux_loss_episode_by_update_by_seed = None

							# Save to data dictionary
							data_dict[plot_update_step] = {
								"rewards_episode_by_seed":temp_rewards_episode_by_seed,
								"actor_enc_dists_step_by_seed":temp_actor_enc_dists_step_by_seed,
								"shared_enc_dists_step_by_seed":temp_shared_enc_dists_step_by_seed,
								"aux_loss_episode_by_update_by_seed":temp_aux_loss_episode_by_update_by_seed,
							}

						# Plot data
						plt.rcParams.update({'font.size':16})
						x_label_pad = 14
						y_label_pad = 16
						for smooth_rewards in [True, False]:
							rewards_min_y = float("inf")
							rewards_max_y = float("-inf")
							fig, axes = plt.subplots(2, 2, figsize=(16, 8))
							fig.subplots_adjust(hspace=0.35, wspace=0.25)
							for plot_update_step in plot_update_steps:

								# Get data for plot update step
								plot_data = data_dict[plot_update_step]
								rewards_episode_by_seed = plot_data["rewards_episode_by_seed"]
								actor_enc_dists_step_by_seed = plot_data["actor_enc_dists_step_by_seed"]
								shared_enc_dists_step_by_seed = plot_data["shared_enc_dists_step_by_seed"]
								aux_loss_episode_by_update_by_seed = plot_data["aux_loss_episode_by_update_by_seed"]

								# Calculate average reward and standard deviation across seeds
								rewards_episode_by_avg_seed = np.mean(rewards_episode_by_seed, axis=1)
								rewards_episode_by_stddev_seed = np.std(rewards_episode_by_seed, axis=1)

								# Calculate median reward and 80% Monte Carlo CI across seeds
								rewards_episode_by_median_seed = np.median(rewards_episode_by_seed, axis=1)
								n_seeds = rewards_episode_by_seed.shape[1]
								lower_bound_seed_idx = round(n_seeds * 0.1)
								upper_bound_seed_idx = round(n_seeds * 0.9)
								rewards_episode_by_seed_sorted = np.sort(rewards_episode_by_seed, axis=1)
								rewards_episode_lower_bound = rewards_episode_by_seed_sorted[:, lower_bound_seed_idx]
								rewards_episode_upper_bound = rewards_episode_by_seed_sorted[:, upper_bound_seed_idx]

								# Calculate average embedding distances across seeds
								actor_enc_dists_step_by_avg_seed = np.mean(actor_enc_dists_step_by_seed, axis=1)
								shared_enc_dists_step_by_avg_seed = np.mean(shared_enc_dists_step_by_seed, axis=1)

								# Calculate average auxiliary task loss across seeds and steps
								if plot_update_step != 0:
									aux_loss_episode_by_update_by_avg_seed = np.mean(
										aux_loss_episode_by_update_by_seed, axis=-1,
									)
									aux_loss_episode_by_avg_update_by_avg_seed = np.mean(
										aux_loss_episode_by_update_by_avg_seed, axis=-1
									)

								# Update plotting parameters
								n_episodes = len(rewards_episode_by_avg_seed)
								n_steps = len(actor_enc_dists_step_by_avg_seed)
								rewards_min_y = min(rewards_min_y, np.min(rewards_episode_by_avg_seed))
								rewards_max_y = max(rewards_max_y, np.max(rewards_episode_by_avg_seed))
								plot_color = color_dict[plot_update_step]

								# Smooth time series
								actor_enc_dists_step_by_avg_seed = gaussian_filter1d(
									actor_enc_dists_step_by_avg_seed,
									enc_dist_gaussian_filter_stddev,
								)
								shared_enc_dists_step_by_avg_seed = gaussian_filter1d(
									shared_enc_dists_step_by_avg_seed,
									enc_dist_gaussian_filter_stddev,
								)
								if plot_update_step != 0:
									aux_loss_episode_by_avg_update_by_avg_seed = gaussian_filter1d(
										aux_loss_episode_by_avg_update_by_avg_seed,
										aux_loss_gaussian_filter_stddev,
									)
								if smooth_rewards:
									rewards_episode_by_avg_seed = gaussian_filter1d(
										rewards_episode_by_avg_seed,
										rewards_gaussian_filter_stddev,
									)

								# Plot rewards
								sns.lineplot(
									x=range(n_episodes),
									y=rewards_episode_by_avg_seed,
									label=plot_update_step,
									ax=axes[0, 0],
									color=plot_color,
									legend=False,
								)
								if plot_CI:
									axes[0, 0].fill_between(
										x=range(n_episodes),
										y1=rewards_episode_lower_bound,
										y2=rewards_episode_upper_bound,
										alpha=0.2,
										color=plot_color,
									)
								if plot_stddev:
									axes[0, 0].fill_between(
										x=range(n_episodes),
										y1=rewards_episode_by_avg_seed - rewards_episode_by_stddev_seed,
										y2=rewards_episode_by_avg_seed + rewards_episode_by_stddev_seed,
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
										ax=axes[1, 0],
										legend=False,
										color=plot_color,
									)
								
								# Plot embedding distances
								sns.lineplot(
									x=range(n_steps),
									y=actor_enc_dists_step_by_avg_seed,
									label=plot_update_step,
									ax=axes[0, 1],
									legend=False,
									color=plot_color,
								)

								# Adjust shared encoder embeddings scale
								if sub_dir == "reset_none":
									plot_shared_enc_dists = shared_enc_dists_step_by_avg_seed / 1.0e+9
									plot_shared_enc_dists_suffix = "(billions)"
								else:
									plot_shared_enc_dists = shared_enc_dists_step_by_avg_seed / 1.0e+6
									plot_shared_enc_dists_suffix = "(millions)"

								sns.lineplot(
									x=range(n_steps),
									y=plot_shared_enc_dists,
									label=plot_update_step,
									ax=axes[1, 1],
									legend=False,
									color=plot_color,
								)

							# Format plot
							axes[0, 0].set_ylim(0, int(rewards_max_y * 1.05))
							axes[0, 0].set_ylabel("Avg Reward", labelpad=y_label_pad)
							axes[1, 0].set_ylabel("Avg Auxiliary Loss", labelpad=y_label_pad)
							axes[0, 1].set_ylabel("Avg Actor Embed \nL2 Distance", labelpad=y_label_pad)
							axes[1, 1].set_ylabel(f"Avg Shared Embed \nL2 Distance {plot_shared_enc_dists_suffix}", labelpad=y_label_pad)
							for i in range(axes.shape[0]):
								axes[i, 0].set_xlabel("Episode", labelpad=x_label_pad)
								axes[i, 1].set_xlabel("Env Step", labelpad=x_label_pad)
								axes[i, 1].xaxis.set_major_formatter(
									ticker.StrMethodFormatter("{x:,.0f}")
								)
								axes[i, 1].yaxis.set_major_formatter(
									ticker.StrMethodFormatter("{x:,.0f}")
								)
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
									f"\nMode: {color_mode}; # seeds: {n_seeds}"
								)
							fig.align_ylabels()
							suffix = "_smoothed_rewards" if smooth_rewards else ""
							fig_name = f"avg_episode_reward_{domain.title()}_{task.title()}_mode_{color_mode}_{n_seeds}_seeds{suffix}.png"
							save_path = os.path.join(results_dir, fig_name)
							fig.savefig(save_path, bbox_inches="tight")

	# Print start and end time
	end_time = time.strftime('%Y-%m-%d %H:%M:%S')
	print(f"Start time: {start_time} \nEnd time: {end_time}")
