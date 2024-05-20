import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

data_folder = "./exp-data/"

algos = ["rr1", "sasr", "rr001"]
envs = ['AntStand', 'AntFar', 'HumanKeep', 'HumanStand', "RobotReach", "RobotPush"]

labels = ['retention rate=1', 'retention rate=0.1(default)', 'retention rate=0.01']
colors = ["#13679E", "#AB3A29", "#D07F2C"]

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

fig.legend(legend_handles, legend_labels, loc='lower center', ncol=3, fontsize=20, columnspacing=1)
plt.subplots_adjust(bottom=0.22, hspace=0.1)

plt.savefig("./diff-retention-rate.svg", bbox_inches='tight', pad_inches=0.05)

plt.show()
