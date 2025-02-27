import os
import time

import pandas as pd
import numpy as np
import torch


if __name__ == "__main__":

	save_tables_dir = "tables"
	os.makedirs(save_tables_dir, exist_ok=True)

	header_format_dict = {
		"cartpole_swingup":"Cartpole Swingup",
		"cheetah_run": "Cheetah Run",
		"color_easy": "Easy Color Shift",
		"color_hard": "Hard Color Shift",
		"inv": "Inverse Dynamics Auxiliary",
		"rot": "Rotation Auxiliary",
		"reset_episode": "Reset Every Episode",
		"reset_none": "No Reset", 
		"reset_ss_updates": "Reset Every Action Step",
	}

	main_tasks = ["cartpole_swingup", "cheetah_run"]
	color_modes = ["color_easy", "color_hard"]
	aux_tasks = ["inv", "rot"]
	reset_agent_modes = ["reset_episode", "reset_none", "reset_ss_updates"]
	aux_update_steps = [0, 1, 2, 4, 8, 16]

	start_time = time.strftime('%Y-%m-%d %H:%M:%S')
	for main_task in main_tasks:
		for color_mode in color_modes:

			all_data_dict = {}
			for aux_task in aux_tasks:

				# Load source domain evaluation baseline
				temp_data_dir = f"logs/{main_task}/{aux_task}/0/eval_train_env"
				temp_data_file_name = f"formatted_data_{main_task}_{aux_task}_eval_train_env_rewards.p"
				with open(os.path.join(temp_data_dir, temp_data_file_name), "rb") as io:
					source_domain_rewards_episode_by_seed = torch.load(io, weights_only=False)
				source_domain_rewards_avg = np.mean(source_domain_rewards_episode_by_seed)
				source_domain_rewards_stddev = np.std(source_domain_rewards_episode_by_seed)

				# Load aggregated data for each reset mode
				for reset_mode in reset_agent_modes:
					temp_data_dir = f"logs/{main_task}/{aux_task}/0/{reset_mode}"
					temp_data_file_name = f"formatted_data_{main_task}_{aux_task}_{reset_mode}_{color_mode}.p"
					with open(os.path.join(temp_data_dir, temp_data_file_name), "rb") as io:
						temp_data = torch.load(io, weights_only=False)

					# Calculate average and stddev reward across seeds and episodes for each aux step
					aux_step_data_dict = {
						"Source Domain":f"{int(source_domain_rewards_avg)} ({int(source_domain_rewards_stddev)})"
					}
					for aux_step in aux_update_steps:
						aux_step_rewards_episode_by_seed = temp_data[aux_step]["rewards_episode_by_seed"]
						aux_step_rewards_avg = np.mean(aux_step_rewards_episode_by_seed)
						aux_step_rewards_stddev = np.std(aux_step_rewards_episode_by_seed)
						aux_step_data_dict[aux_step] = f"{int(aux_step_rewards_avg)} ({int(aux_step_rewards_stddev)})"
					
					# Save data series with table header
					formatted_table_header = (
						header_format_dict[aux_task],
						header_format_dict[reset_mode],
					)
					all_data_dict[formatted_table_header] = aux_step_data_dict

			# Create LaTeX table
			table_headers = pd.MultiIndex.from_tuples(all_data_dict.keys())
			df = pd.DataFrame(all_data_dict.values(), index=table_headers).T
			latex_table = df.to_latex(index=True, escape=False, multirow=True)

			# Save LaTeX table
			table_file = f"results_table_{main_task}_{color_mode}.txt"
			with open(os.path.join(save_tables_dir, table_file), "w") as io:
				io.write(latex_table)
