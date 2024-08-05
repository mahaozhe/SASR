import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

data_folder = "./exp-data/"

algos = ["m50", "m500", "sasr", "m2000"]
envs = ['AntStand', 'AntSpeed', 'AntFar', 'AntVeryFar', 'WalkerKeep',
        'HumanStand', 'HumanKeep', 'RobotReach', 'RobotPush', 'MountainCar']

labels = [r"$M=50$", r"$M=500$", r"$M=1000$ (default)", r"$M=2000$"]
colors = ["#1E7C4A", "#D07F2C", "#AB3A29", "#13679E"]

fig, axs = plt.subplots(2, 5, figsize=(25, 6))

for i in range(len(envs)):
    if i == 0:
        axs[i // 5][i % 5].set_ylabel('Episode returns', fontsize=20)
    if i == 5:
        axs[i // 5][i % 5].set_ylabel('Episode returns', fontsize=20)
        axs[i // 5][i % 5].set_xlabel(r'Steps ($\times 10^3$)', fontsize=20)
        # axs[i // 5][i % 5].tick_params(axis='both', which='both')

    print(f"=========={envs[i]}==========")

    for j in range(len(algos)):
        data_path = os.path.join(data_folder, envs[i], f"{algos[j]}.npy")

        data_steps, data_mean, data_std = np.load(data_path)

        axs[i // 5][i % 5].fill_between(np.array(data_steps) / 1000,
                                        data_mean + data_std,
                                        data_mean - data_std,
                                        alpha=0.3, color=colors[j], label=labels[j])

        axs[i // 5][i % 5].plot(np.array(data_steps) / 1000, data_mean, color=colors[j])

        # compute the mean and the std error
        print(f"{envs[i]}: {labels[j]}: {np.mean(data_mean):.2f} +- {np.std(data_mean) / np.sqrt(len(data_mean)):.2f}")

    axs[i // 5][i % 5].set_title(envs[i], fontsize=24)

# get the legend from the first sub-figure
legend_handles, legend_labels = axs[0][0].get_legend_handles_labels()
# # reorder the legend
# order = [8, 7, 6, 5, 4, 3, 2, 1, 0]
# handles_new = [legend_handles[i] for i in order]
# labels_new = [legend_labels[i] for i in order]

fig.legend(legend_handles, legend_labels, loc='lower center', ncol=9, fontsize=17, columnspacing=1)
plt.subplots_adjust(bottom=0.14, hspace=0.3)

plt.savefig("./rff-dim-new.pdf", bbox_inches='tight', pad_inches=0.05)

plt.show()
