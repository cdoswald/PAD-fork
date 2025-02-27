import numpy as np
import torch
import os
import utils
from video import VideoRecorder

from arguments import parse_args
from agent.agent import make_agent
from eval import init_env, evaluate


def main(args):
	# Initialize environment
	env = init_env(args)

	# Create dirs
	model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	cropped_obs_shape = (3*args.frame_stack, 84, 84)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)
	agent.load(model_dir, args.pad_checkpoint)

	# Evaluate agent on original training environment (no domain shift)
	print(
		f'Evaluating {args.work_dir} for {args.pad_num_episodes} episodes '+
		f'(mode: {args.mode}; eval seed: {args.seed}'
	)
	episode_rewards, _, _ = evaluate(env, agent, args, video, adapt=False)
	print('Average episode reward:', round(np.mean(episode_rewards), 0))

	# Save results
	subdir = os.path.join(args.work_dir, f"eval_train_env")
	os.makedirs(subdir, exist_ok=True)
	results_fp = os.path.join(subdir, f'pad_{args.mode}_evalseed_{args.seed}.pt')
	torch.save({
		'args': args,
		'episode_rewards': episode_rewards,
	}, results_fp)
	print('Saved results to', results_fp)
	

if __name__ == '__main__':
	args = parse_args()
	main(args)
