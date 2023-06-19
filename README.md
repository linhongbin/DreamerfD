# Efficient-Dreamer: Integration of World Models and Behavior Cloning

## Overview

We integrate [Dreamerv2](https://github.com/danijar/dreamerv2.git) with Behavior Cloning, where a set of off-policy demonstration trajectory is ultilized to boost the learning efficiency of world model learning.

## Method 

Compared with Dreamerv2, we have the following improvements/novelties:

- Demonstration-guided wolrd model learning and policy learning:
    Additional off-policy demonstration data is sampled by Replay Buffer for world model learning and behavior learning.
<p style="text-align:center;">
    <img src="doc/media/DreamerBC_1.png" width=270 title="xxx" class="center">
    <img src="doc/media/DreamerBC_2.png" width=250>
</p>

- Behaviour cloning loss to regress actions of demonstration trajectories.
  <p style="text-align:center;">
    <img src="doc/media/DreamerBC_3.png" width=270>
</p>

For more information:
- [Project Website](https://sites.google.com/view/dreamerbc/home)
## Feature

- Benchmark on [AccelNet Surgical Challenge](https://github.com/collaborative-robotics/surgical_robotics_challenge)
- Support multiple needle variations
- Integration with [gym](https://github.com/openai/gym), [dVRK](https://github.com/jhu-dvrk)


## Install

- Install [Anaconda](https://www.anaconda.com/download)

- Install Dependencies with GPU support
    ```sh
    conda create -n efficient_dreamer python=3.7
    conda activate efficient_dreamer
    conda install cudatoolkit=11.3 -c pytorch
    pip install tensorflow==2.9.0 tensorflow_probability==0.17.0
    conda install cudnn=8.2 -c anaconda
    pip install protobuf==3.20.1
    ```
- Install Efficient-Dreamer
    ```sh
    pip install -e .
    ```

## Run
  For all command lines, we assume the current directory is `<path to gym-suture>`, otherwise, change directory by
  ```sh
  cd <path to gym-suture>
  ```
