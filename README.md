# Self-Adaptive Success Rate based Reward Shaping (SASR)

## Table of Contents

- [Requirements](#requirements)
- [Run SASR Algorithm](#run-sasr-algorithm)
- [Some Experimental Results](#some-experimental-results)

## Requirements

- This code has been tested on:
```
pytorch==2.0.1+cu117
```
- Install all dependent packages:
```
pip3 install -r requirements.txt
```

## Run SASR Algorithm

Run the following command to run SASR algorithm on the task specified by `<Task ID>`:

```
python run-SASR.py --env-id <Task ID>
```

All available environments with sparse rewards evaluated in our paper are listed below:

![All available environments with sparse rewards](./readme-images/environments.png)

* Mujoco-Sparse:
    - `MyMujoco/Ant-Height-Sparse`: the *AntStand* task.
    - `MyMujoco/Ant-Far-Sparse`: the *AntFar* task.
    - `MyMujoco/Humanoid-Keep-Sparse`: the *HumanKeep* task.
    - `MyMujoco/HumanoidStandup-Sparse`: the *HumanStand* task.
* Robotics-Sparse:
    - `MyFetchRobot/Reach-Jnt-Sparse-v0`: the *RobotReach* task.
    - `MyFetchRobot/Push-Jnt-Sparse-v0`: the *RobotPush* task.
* Classic control:
    - `MountainCarContinuous-v0`: the *MountainCar* task.

All hyper-parameters are set as default values in the code. You can change them by adding arguments to the command line. All available arguments are listed below:

```
--env-id: the task id
--exp-name: the name of the experiment, to record the tensorboard and save the model.
--seed: the random seed.
--cuda: the cuda device, default is 0, the code will automatically choose "cpu" if cuda is not available.
--gamma: the discount factor.

--proposed-reward-scale: the scale of the proposed reward, default is 1.
--beta: the weight of the proposed reward, default is 0.2.

--pa-buffer-size: the buffer size of the policy agent.
--pa-rb-optimize-memory: whether to optimize the memory of the policy agent
--pa-batch-size: the batch size of the policy agent
--ra-buffer-size: the buffer size of the reward agent.
--ra-rb-optimize-memory: whether to optimize the memory of the reward agent
--ra-batch-size: the batch size of the reward agent

--pa-actor-lr: the learning rate of the actor of the policy agent
--pa-critic-lr: the learning rate of the critic of the policy agent
--pa-alpha-lr: the learning rate of the alpha of the policy agent
--ra-actor-lr: the learning rate of the actor of the reward agent
--ra-critic-lr: the learning rate of the critic of the reward agent
--ra-alpha-lr: the learning rate of the alpha of the reward agent

--pa-policy-frequency: the policy frequency of the policy agent
--pa-target-frequency: the target frequency of the policy agent
--pa-tau: the tau of the policy agent
--ra-policy-frequency: the policy frequency of the reward agent
--ra-target-frequency: the target frequency of the reward agent
--ra-tau: the tau of the reward agent

--pa-alpha: the alpha of the policy agent
--pa-alpha-autotune: whether to autotune the alpha of the policy agent
--ra-alpha: the alpha of the reward agent
--ra-alpha-autotune: whether to autotune the alpha of the reward agent

--write-frequency: the frequency to write the tensorboard
--save-folder: the folder to save the model
```

## Some Experimental Results

The saved experimental results can be found in [this folder](./experiments/data). You can run the following command to show the experimental results:

- To evaluate the learning performance in comparison with baselines:
```
python ./experiments/comparison.py
```

![Comparison the learning performance of ReLara with the baselines.](./readme-images/comparison-baselines.svg)


- To compare ReLara with learning R(s,a) and R(s) as reward functions:

```
python ./experiments/rsa-vs-rs.py
```

![Comparison of the ReLara with the reward function R(s, a) and its variant to learn the R(s) function.](./readme-images/rsa-vs-rs.svg)

- To compare ReLara with two ablation variants, one with the reward agent at pre-half-stage and another with the reward agent at post-half-stage:

```
python ./experiments/pre-vs-post-stages.py
```

![Comparison of ReLara with the two variants that the reward agent is only involved in the pre- and post-half stages.](./readme-images/pre-vs-post-stages.svg)

- To compare ReLara with different scales of the proposed reward:

```
python ./experiments/diff-beta.py
```

![Comparison of different suggested reward weight factors in ReLara.](./readme-images/diff-beta.svg)



