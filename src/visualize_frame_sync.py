from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import torch


def show_all_frames(data):
    eval_frames = data[0]
    orig_frames = data[1]
    n_frames = len(eval_frames)
    fig, axes = plt.subplots(2, n_frames, figsize=(8,6))
    fig.subplots_adjust(hspace=-0.05)
    for i, eval_frame in enumerate(eval_frames):
        axes[0, i].imshow(eval_frame.transpose(1, 2, 0))
    for j, orig_frame in enumerate(orig_frames):
        axes[1, j].imshow(orig_frame.transpose(1, 2, 0))
    axes[0, 0].set_title("Evaluation Environment")
    axes[1, 0].set_title("Original Environment")
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    return fig


if __name__ == "__main__":

    color_dict = {
        0:"tab:blue",
        1:"tab:orange",
        2:"tab:green",
        4:"tab:red",
        8:"tab:purple",
        16:"tab:brown"
    }

    work_dir = "logs/cheetah_run/inv/0"
    frames_sync_dir = os.path.join(work_dir, "frame_sync_validation")
    frames_sync_files = glob(os.path.join(frames_sync_dir, "frames_sync*"))
    figures_dir = os.path.join(frames_sync_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for file in frames_sync_files[::2]:
        with open(file, "rb") as io:
            data = pickle.load(io)
        fig = show_all_frames(data)
        save_path = os.path.join(figures_dir, f"{Path(file).stem}.png")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close()