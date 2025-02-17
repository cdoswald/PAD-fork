import os

import matplotlib.pyplot as plt

from env.wrappers import make_pad_env
import utils


if __name__ == "__main__":

    seed = 0
    savedir = "images"

    utils.set_seed_everywhere(seed)
    train_env = make_pad_env(
        domain_name="cheetah",
        task_name="run",
        seed=seed,
        episode_length=1000,
        action_repeat=4,
        mode="train",
    )

    utils.set_seed_everywhere(seed)
    color_easy_env = make_pad_env(
        domain_name="cheetah",
        task_name="run",
        seed=seed,
        episode_length=1000,
        action_repeat=4,
        mode="color_easy",
    )

    utils.set_seed_everywhere(seed)
    color_hard_env = make_pad_env(
        domain_name="cheetah",
        task_name="run",
        seed=seed,
        episode_length=1000,
        action_repeat=4,
        mode="color_hard",
    )

    train_obs = train_env.reset()
    color_easy_obs = color_easy_env.reset()
    color_hard_obs = color_hard_env.reset()

    fig, axes = plt.subplots(1, 3, figsize=(8,6))
    axes[0].imshow(train_obs[:3, :, :].transpose(1, 2, 0))
    axes[1].imshow(color_easy_obs[:3, :, :].transpose(1, 2, 0))
    axes[2].imshow(color_hard_obs[:3, :, :].transpose(1, 2, 0))
    for i in range(len(axes)):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.savefig(os.path.join(savedir, "train_color_easy_hard_domain_shifts.png"), bbox_inches="tight")
