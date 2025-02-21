import os
import time

import pandas as pd
import numpy as np
import torch


if __name__ == "__main__":

	steps_per_episode = 1000
	aux_update_steps = [0, 1, 2, 4, 8, 16]

	color_modes = ["color_easy", "color_hard"]
	main_tasks = ["cartpole_swingup", "cheetah_run"]
	aux_tasks = ["inv", "rot"]
	reset_agent_modes = ["reset_episode", "reset_none", "reset_ss_updates"]

	start_time = time.strftime('%Y-%m-%d %H:%M:%S')
	for color_mode in color_modes:

		# Create blank dataframe
		table_headers = [
			(i.title().replace("_", " "), j.title(), k.title()) 
			for i in main_tasks
			for j in aux_tasks
			for k in reset_agent_modes
		]
		table_cols = pd.MultiIndex.from_tuples(table_headers)

		for main_task in main_tasks:
			for aux_task in aux_tasks:
				for reset_mode in reset_agent_modes:

					# Load aggregated data
					temp_data_dir = f"logs/{main_task}/{aux_task}/0/{reset_mode}"
					temp_data_file_name = f"formatted_data_{main_task}_{aux_task}_{reset_mode}_{color_mode}.p"
					with open(os.path.join(temp_data_dir, temp_data_file_name), "rb") as io:
						temp_data = torch.load(io)

					# Calculate average and stddev reward across seeds and episodes for each aux step
					for aux_step in aux_update_steps:
						aux_step_rewards_episode_by_seed = temp_data[aux_step]["rewards_episode_by_seed"]
						aux_step_rewards_avg = np.mean(aux_step_rewards_episode_by_seed)
						aux_step_rewards_stddev = np.std(aux_step_rewards_episode_by_seed)

