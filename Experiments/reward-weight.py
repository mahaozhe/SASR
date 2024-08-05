import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

data_folder = "./exp-data/"

algos = ["rw02", "rw04", "sasr", "rw08", "rw10"]
envs = ['AntStand', 'AntFar', 'HumanStand', 'HumanKeep', "RobotReach", "RobotPush"]

labels = [r'$\lambda=0.2$', r'$\lambda=0.4$', r'$\lambda=0.6$ (default)', r'$\lambda=0.8$', r'$\lambda=1.0$']
colors = ["#13679E", "#D07F2C", "#AB3A29", "#1E7C4A", "#6F6DA1"]

fig, axs = plt.subplots(1, 6, figsize=(30, 4.5))

for i in range(len(envs)):
    if i == 0:
        axs[i].set_ylabel('Episode returns', fontsize=20)
        axs[i].set_xlabel(r'Steps ($\times 10^3$)', fontsize=20)

    for j in range(len(algos)):
        data_path = os.path.join(data_folder, envs[i], f"{algos[j]}.npy")

        data_steps, data_mean, data_std = np.load(data_path)

        axs[i % 6].fill_between(np.array(data_steps) / 1000,
                                data_mean + data_std,
                                data_mean - data_std,
                                alpha=0.3, color=colors[j], label=labels[j])
        axs[i % 6].plot(np.array(data_steps) / 1000, data_mean, color=colors[j])

    axs[i % 6].set_title(envs[i], fontsize=28)

# get the legend from the first sub-figure
legend_handles, legend_labels = axs[0].get_legend_handles_labels()

fig.legend(legend_handles, legend_labels, loc='lower center', ncol=5, fontsize=20, columnspacing=1)
plt.subplots_adjust(bottom=0.24, hspace=0.1)

plt.savefig("./diff-reward-weights.pdf", bbox_inches='tight', pad_inches=0.05)

plt.show()
