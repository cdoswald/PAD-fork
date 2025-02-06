import numpy as np
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
	embedding_dists = {"actor_enc":[], "aux_encoder":[]}

	# Create common seed for resetting evaluation and original environments
	first_reset_random_seed = int(np.random.random() * 9999)
	print(f"First reset random seed: {first_reset_random_seed}")

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

		env.seed(first_reset_random_seed)
		obs = env.reset()
		done = False
		episode_reward = 0
		losses = []
		step = 0
		ep_agent.train()

		if orig_env is not None:
			orig_env.seed(first_reset_random_seed)
			orig_obs = orig_env.reset()
			assert np.all(orig_env.get_state() == env.get_state())

		# Start evaluation
		while not done:

			# Take step
			with utils.eval_mode(ep_agent):
				action = ep_agent.select_action(obs)
			next_obs, reward, done, _ = env.step(action)
			episode_reward += reward

			# Take EXACT same step with orig environment to compare embeddings (if applicable)
			if orig_env is not None:
				orig_next_obs, _, _, _ = orig_env.step(action)
				assert np.all(orig_env.get_state() == env.get_state())
				assert orig_next_obs.shape == next_obs.shape, print(
					f"orig next obs shape: {orig_obs.shape}; eval next obs shape: {obs.shape}"
				)

				# Convert to batch format
				orig_next_obs_batch = utils.batch_from_obs(
					torch.Tensor(orig_next_obs).cuda(), batch_size=args.pad_batch_size
				)
				next_obs_batch = utils.batch_from_obs(
					torch.Tensor(next_obs).cuda(), batch_size=args.pad_batch_size
				)

				# Calculate distance between original and eval obs embeddings 
				orig_enc_actor = orig_agent.actor.encoder(orig_next_obs_batch, detach=True).detach().cpu()
				updated_enc_actor = ep_agent.actor.encoder(next_obs_batch, detach=True).detach().cpu()

				orig_enc_aux = orig_agent.ss_encoder(orig_next_obs_batch, detach=True).detach().cpu()
				updated_enc_aux = ep_agent.ss_encoder(next_obs_batch, detach=True).detach().cpu()

				actor_enc_dist = torch.norm(orig_enc_actor - updated_enc_actor, p=2) # L2 norm
				aux_enc_dist = torch.norm(orig_enc_aux - updated_enc_aux, p=2)

				embedding_dists["actor_enc"].append(actor_enc_dist)
				embedding_dists["aux_encoder"].append(aux_enc_dist)

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



# ## Debug
# import matplotlib.pyplot as plt

# def show_frame(input: np.ndarray, frame=1):
# 	plt.imshow(input[(3*(frame-1)):3*(frame), :, :].transpose(1, 2, 0))

# def init_env(args):
# 		utils.set_seed_everywhere(args.seed)
# 		return make_pad_env(
# 			domain_name=args.domain_name,
# 			task_name=args.task_name,
# 			seed=args.seed,
# 			episode_length=args.episode_length,
# 			action_repeat=args.action_repeat,
# 			mode=args.mode
# 		)

# colors = torch.load(f"src/env/data/color_easy.pt")

# class TestArgs():

# 	def __init__(self):
# 		pass


# args = TestArgs()
# args.domain_name = "cartpole"
# args.task_name = "swingup"
# args.seed = 1
# args.episode_length = 1000
# args.action_repeat = 8
# args.mode = "color_hard"

# env = make_pad_env(
# 	args.domain_name,
# 	args.task_name,
# 	args.seed,
# 	args.episode_length,
# 	action_repeat=args.action_repeat,
# 	mode=args.mode
# )



# default_colors = {
# 	'grid_rgb1': np.array([0.1, 0.2, 0.3]),
# 	'grid_rgb2': np.array([0.2, 0.3, 0.4]),
# 	'self_rgb': np.array([0.7, 0.5, 0.3]), 
# 	'skybox_rgb': np.array([0.4, 0.6, 0.8]),
# }

# obs = env.reset()
# show_frame(obs, 3)

# dir(env.physics.model)

# env.unwrapped.physics.named.model.geom_rgba

# env.unwrapped.physics.named.model.mat_rgba

# env.unwrapped.physics.named.model.site_rgba
# env.unwrapped.physics.named.model.skin_rgba
# env.unwrapped.physics.named.model.tendon_rgba
# env.unwrapped.physics.named.model.tex_rgb




# env.physics.model.geom_rgba

# env.reload_physics(default_colors)
# env.unwrapped.render("rgb_array")
# show_frame(env.env._get_obs(), 3)


# env.reload_physics()
# obs = env.reset()
# plt.imshow(obs[:3, :, :].transpose(1, 2, 0))



# obs_standard = env_standard.reset()
# show_frame(obs_standard, frame=1)

# exp_state = env.get_state()
# show_frame(obs, frame=1)


# env_standard.set_state(env.get_state())
# assert np.all(env_standard.get_state() == env.get_state())
# action = np.ndarray(shape=(1,), dtype=np.float32) * 4

# next_obs, _, _, _ = env.step(action)
# next_obs_standard, _, _, _ = env_standard.step(action)
# show_frame(next_obs, 3)
# show_frame(next_obs_standard, 3)

# obs = next_obs




# env_standard.set_state(env.get_state())
# assert np.all(env_standard.get_state() == env.get_state())

# # Get original env obs rendering for comparison with eval env obs

# orig_obs = env_standard._get_dmc_wrapper().render()
# assert orig_obs.shape == obs.shape, print(
# 	f"orig obs shape: {orig_obs.shape}; eval obs shape: {obs.shape}"
# )





# DEL = env_standard.reset()
# show_frame(DEL, 3)
# env_state = env.get_state()
# env_standard.set_state(env_state + [4, -8, 0, 0])
# env_standard.get_state()
# new_obs = env_standard.env._get_obs()
# show_frame(new_obs, 3)

# new_frame = env_standard.unwrapped.render().transpose(2, 0, 1)
# show_frame(new_frame)


# env.unwrapped.current_state 
# env_standard.unwrapped.current_state

# env_standard.unwrapped.current_state = deepcopy(env.unwrapped.current_state) + [4, -8, 1, 10, 16]
# new_frame = env_standard.unwrapped.render().transpose(2, 0, 1)
# show_frame(new_frame)




# env_standard.env._frames.append(env_standard.unwrapped.render().transpose(2, 0, 1))
# len(env_standard.env._frames)
# show_frame(env_standard.env._frames[2])

# train_physics = env_standard.physics
# eval_physics = env.physics

# train_physics.data.qpos[:] = eval_physics.data.qpos[:]
# train_physics.data.qvel[:] = eval_physics.data.qvel[:]
# train_physics.step()
# env_standard.unwrapped.current_state


# env_standard.unwrapped._env.physics is env_standard.physics


# env_standard.unwrapped._get_obs()
# pixels = train_physics.render(height=100, width=100, camera_id=0)
# env_standard.render(mode="rgb_array").shape
# show_frame(pixels)