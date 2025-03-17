import numpy as np
import pickle
import torch
import os
from copy import deepcopy
from tqdm import tqdm
import utils
from video import VideoRecorder

from arguments import parse_args
from env.wrappers import make_pad_env
from agent.agent import make_agent
from utils import get_curl_pos_neg


def evaluate(env, agent, args, video, adapt=False, orig_env=None):
	"""Evaluate an agent, optionally adapt using PAD"""
	episode_rewards = []
	all_losses = []
	embedding_dists = {"actor_enc":[], "aux_enc":[], "shared_conv_enc":[]}

	validate_frame_sync_iter = 0
	validate_frame_sync_dir = os.path.join(args.work_dir, "frame_sync_validation")
	os.makedirs(validate_frame_sync_dir, exist_ok=True)
	
	# Make copy of agent to compare train and eval embeddings (if applicable)
	if orig_env is not None:
		orig_agent = deepcopy(agent)

	for i in tqdm(range(args.pad_num_episodes)):

		if i == 0 or args.pad_reset_agent == "episode":
			ep_agent = deepcopy(agent) # make a new copy to update with aux task

		if args.use_curl: # initialize replay buffer for CURL
			replay_buffer = utils.ReplayBuffer(
				obs_shape=env.observation_space.shape,
				action_shape=env.action_space.shape,
				capacity=args.train_steps,
				batch_size=args.pad_batch_size
			)
		video.init(enabled=True)

		obs = env.reset()
		done = False
		episode_reward = 0
		losses = []
		step = 0
		ep_agent.train()

		# Freeze episode agent auxiliary decoder (if applicable)
		if args.freeze_aux_decoder:
			if i == 0:
				print("Freezing auxiliary decoder")
			if args.use_inv:
				for param in ep_agent.inv.parameters():
					param.requires_grad = False
			elif args.use_rot:
				for param in ep_agent.rot.parameters():
					param.requires_grad = False
			else:
				raise Warning("Freeze auxiliary decoder set to True, but no auxiliary model specified")
		
		# Freeze original agent auxiliary decoder (if applicable)
		if args.freeze_aux_decoder and orig_env is not None:
			if args.use_inv:
				for param in orig_agent.inv.parameters():
					param.requires_grad = False
			elif args.use_rot:
				for param in orig_agent.rot.parameters():
					param.requires_grad = False

		# Synchronize starting position
		# (note that we have to do this here since the initial frame stack queue starts
		# with 3 observations, and only 1 observation is added in subsequent steps)
		if orig_env is not None:
			_ = orig_env.reset()
			orig_env.physics.data.qpos[:] = deepcopy(env.physics.data.qpos[:])
			orig_env.physics.forward()
			assert np.all(orig_env.get_state()[:2] == env.get_state()[:2]) # check position

			# Reset initial frame stack queue after matching state
			orig_env_obs = orig_env.render("rgb_array").transpose(2, 0, 1)
			for _ in range(len(orig_env.env._frames)): # starts with stack of k=3 frames
				orig_env.env._frames.append(orig_env_obs)

		# Start evaluation
		while not done:

			# Take step
			with utils.eval_mode(ep_agent):
				action = ep_agent.select_action(obs)
			next_obs, reward, done, _ = env.step(action)
			episode_reward += reward

			# Calculate embedding distance between train and eval environments (if applicable)
			if orig_env is not None:

				# Update orig env position to match eval env position
				orig_env.physics.data.qpos[:] = deepcopy(env.physics.data.qpos[:])
				orig_env.physics.forward()
				assert np.all(orig_env.get_state()[:2] == env.get_state()[:2]) # check position

				# Render and add observation to frame stack queue
				orig_env.env._frames.append(orig_env.render("rgb_array").transpose(2, 0, 1))

				# Save out frames from original and evaluation environment 
				# periodically to validate that state position matches
				validate_frame_sync_iter += 1
				if validate_frame_sync_iter % 800 == 0:
					temp_file = f"frames_sync_mode_{args.mode}_evalseed_{args.seed}_iter{validate_frame_sync_iter}.p"
					with open(os.path.join(validate_frame_sync_dir, temp_file), "wb") as io:
						pickle.dump((env.env._frames, orig_env.env._frames), io)

				# Convert observation to batch format
				orig_env_next_obs = orig_env.env._get_obs()
				orig_env_next_obs_batch = utils.batch_from_obs(
					torch.Tensor(orig_env_next_obs).cuda(),
					batch_size=args.pad_batch_size
				)
				next_obs_batch = utils.batch_from_obs(
					torch.Tensor(next_obs).cuda(),
					batch_size=args.pad_batch_size
				)

				# Calculate distance between original and evaluation obs embeddings
				# (using all layers in respective encoders, including non-shared conv, linear, and layernorm)
				orig_enc_actor = orig_agent.actor.encoder(orig_env_next_obs_batch, detach=True).detach().cpu()
				updated_enc_actor = ep_agent.actor.encoder(next_obs_batch, detach=True).detach().cpu()
				actor_enc_dist = torch.norm(orig_enc_actor - updated_enc_actor, p=2).item() # L2 norm
				embedding_dists["actor_enc"].append(actor_enc_dist)

				# Calculate distance between original and evaluation obs embeddings
				# (using only the layers shared by all three of the actor, critic, and aux encoders)
				orig_enc_actor_shared_only = orig_agent.actor.encoder.forward_conv_n_layers(
					orig_env_next_obs_batch,
					n_layers=args.num_shared_layers,
					detach=True,
				).detach().cpu()
				updated_enc_actor_shared_only = ep_agent.actor.encoder.forward_conv_n_layers(
					next_obs_batch,
					n_layers=args.num_shared_layers,
					detach=True,
				).detach().cpu()
				actor_enc_dist_shared_only = torch.norm(
					orig_enc_actor_shared_only - updated_enc_actor_shared_only,
					p=2
				).item()
				embedding_dists["shared_conv_enc"].append(actor_enc_dist_shared_only)

			# Reset model weights between environment steps (if applicable)
			if args.pad_reset_agent == "ss_updates":
				ep_agent = deepcopy(agent)
				ep_agent.train()

			# Make self-supervised update if flag is true
			if adapt:
				if args.use_rot: # rotation prediction
					for _ in range(args.ss_update_quantity):

						# Prepare batch of cropped observations
						batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
						batch_next_obs = utils.random_crop(batch_next_obs)

						# Adapt using rotation prediction
						losses.append(ep_agent.update_rot(batch_next_obs))
				
				if args.use_inv: # inverse dynamics model
					for _ in range(args.ss_update_quantity):

						# Prepare batch of observations
						batch_obs = utils.batch_from_obs(torch.Tensor(obs).cuda(), batch_size=args.pad_batch_size)
						batch_next_obs = utils.batch_from_obs(torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size)
						batch_action = torch.Tensor(action).cuda().unsqueeze(0).repeat(args.pad_batch_size, 1)

						# Adapt using inverse dynamics prediction
						losses.append(ep_agent.update_inv(utils.random_crop(batch_obs), utils.random_crop(batch_next_obs), batch_action))

				if args.use_curl: # CURL
					for _ in range(args.ss_update_quantity):

						# Add observation to replay buffer for use as negative samples
						# (only first argument obs is used, but we store all for convenience)
						replay_buffer.add(obs, action, reward, next_obs, True)

						# Prepare positive and negative samples
						obs_anchor, obs_pos = get_curl_pos_neg(next_obs, replay_buffer)

						# Adapt using CURL
						losses.append(ep_agent.update_curl(obs_anchor, obs_pos, ema=True))

			video.record(env, losses)
			obs = next_obs
			step += 1

		reset_suffix = f"reset_{args.pad_reset_agent}"
		video.save(f'{args.mode}_{reset_suffix}_pad_{i}.mp4' if adapt else f'{args.mode}_eval_{i}.mp4')
		episode_rewards.append(episode_reward)
		all_losses.append(losses)

	return episode_rewards, all_losses, embedding_dists


def init_env(args):
		utils.set_seed_everywhere(args.seed)
		return make_pad_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed,
			episode_length=args.episode_length,
			action_repeat=args.action_repeat,
			mode=args.mode
		)


def main(args):
	# Initialize environment
	env = init_env(args)

	# Create original training environment to compare embeddings
	args_copy = deepcopy(args)
	args_copy.mode = "train"
	orig_env = init_env(args_copy)

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

	# Evaluate agent
	ss_update_quantities = [0] 
	if args.use_inv or args.use_curl or args.use_rot:
		ss_update_quantities += [int(x) for x in args.ss_update_quantities.split(",")]

	all_rewards = {}
	all_aux_losses = {}
	all_embed_dists = {}

	for ss_update_quantity in ss_update_quantities:
		args.ss_update_quantity = ss_update_quantity
		env = init_env(args)
		print(
			f'Evaluating {args.work_dir} for {args.pad_num_episodes} episodes '+
			f'(mode: {args.mode}; eval seed: {args.seed}; '+
			f'SSL update steps: {ss_update_quantity})'
		)
		episode_rewards, aux_losses, embed_dists = evaluate(
			env, agent, args, video,
			adapt=(True if ss_update_quantity > 0 else False),
			orig_env=orig_env,
		)
		print('Average episode reward:', round(np.mean(episode_rewards), 0))
		all_rewards[ss_update_quantity] = episode_rewards
		all_aux_losses[ss_update_quantity] = aux_losses
		all_embed_dists[ss_update_quantity] = embed_dists

	# Save results
	subdir = os.path.join(args.work_dir, f"reset_{args.pad_reset_agent}")
	os.makedirs(subdir, exist_ok=True)
	results_fp = os.path.join(subdir, f'pad_{args.mode}_evalseed_{args.seed}.pt')
	torch.save({
		'args': args,
		'episode_rewards': all_rewards,
		'aux_losses': all_aux_losses,
		'embed_dists': all_embed_dists,
	}, results_fp)
	print('Saved results to', results_fp)


if __name__ == '__main__':
	args = parse_args()
	main(args)