"""
The script to plot the comparison of SASR with other baselines.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

data_folder = "./exp-data/"

algos = ["ppo", "td3", "sac", "rnd", "explors", "rosa", "sasr"]
envs = ['AntStand', 'AntFar', 'HumanKeep', 'HumanStand', "RobotReach", "RobotPush"]

labels = ["PPO", "TD3", "SAC", "RND", "ExploRS", "ROSA", "TSRS"]
colors = ["#757574", "#E5A2C4", "#6F6DA1", "#1E7C4A", "#D07F2C", "#13679E", "#AB3A29"]

fig, axs = plt.subplots(1, 6, figsize=(30, 5))

for i in range(len(envs)):
    if i == 0:
        axs[i].set_ylabel('Episode returns', fontsize=20)
        axs[i].set_xlabel('Steps (in thousands)', fontsize=20)
        axs[i].tick_params(axis='both', which='both')

    for j in range(len(algos)):
        data_path = os.path.join(data_folder, envs[i], f"{algos[j]}.npy")

        data_steps, data_mean, data_std = np.load(data_path)

        axs[i % 6].fill_between(np.array(data_steps) / 1000,
                                data_mean + data_std,
                                data_mean - data_std,
                                alpha=0.3, color=colors[j], label=labels[j])
        axs[i % 6].plot(np.array(data_steps) / 1000, data_mean, color=colors[j])

    axs[i % 6].set_title(envs[i], fontsize=24)

# get the legend from the first sub-figure
legend_handles, legend_labels = axs[0].get_legend_handles_labels()
# reorder the legend
order = [6, 5, 4, 3, 2, 1, 0]
handles_new = [legend_handles[i] for i in order]
labels_new = [legend_labels[i] for i in order]

fig.legend(handles_new, labels_new, loc='lower center', ncol=7, fontsize=20, columnspacing=1)
plt.subplots_adjust(bottom=0.22, hspace=0.1)

plt.savefig("./comparison.svg", bbox_inches='tight', pad_inches=0.05)

plt.show()