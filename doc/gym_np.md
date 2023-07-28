## Run
  For all command lines, we assume the current directory is `<path to gym-suture>`, otherwise, change directory by
  ```sh
  cd <path to gym-suture>
  ```
### 1. Launch simulator
- **Launch roscore, open 1st terminal and type**
  ```
  roscore
  ```

- **Launch simulator, open 2st terminal and type**
  ```sh
  source ./bash/ambf/simulator_phantom_rgb.sh # For standard needle
  ```
  > To evaluate different needle variations, type the following commands instead:
  >
  >  ```sh
  >  source ./bash/ambf/simulator_phantom_rgb_small.sh # >For small needle
  >  ```
  >  ```sh
  >  source ./bash/ambf/simulator_phantom_rgb_large.sh # For large needle
  >  ```
  >  ```sh
  >  source ./bash/ambf/simulator_phantom_rgb_unregular.sh # For irregular shape 1
  >  ```
  >  ```sh
  >  source ./bash/ambf/simulator_phantom_rgb_unregular2.sh # For irregular shape 2
  >  ```

- **Launch crtk interface for control, open 3st terminal and type**
  ```sh
  source bash/ambf/crtk.sh
  ```

- **Reset robotic arm to ready position, open 4st terminal and type**
  ```sh
  source bash/ambf/reset.sh 
  ```
### 2. Start Training

- **Train with DreamerfD**
  ```sh
  source bash/ambf/init.sh
  python ext/DreamerfD/example/suture_json.py --default-json ./ext/DreamerfD/example/jsons/suture_np/default_np.yaml --section 1 --logdir ./data/rl_log
  ```
  It will take 3 days on RTX3070 GPU (2 days on RTX 3090)  for convergence of evaluated succeses rate (reaching 80%).
  > Resuming training is supported. To resume training, you need to use different seed number and use the same section number w.r.t. your preivous training, take the previous command for example, you can run
  > ```
  > python ext/DreamerfD/example/suture_json.py --default-json ./ext/DreamerfD/example/jsons/suture_np/default_np.yaml --section 1 --logdir ./data/rl_log --seed 2
  >```


- **Train with other baselines**
  
  Baseline: [Dreamer](https://danijar.com/project/dreamerv2/)
  ```sh
  source bash/ambf/init.sh
  python ext/DreamerfD/example/suture_json.py --default-json ./ext/DreamerfD/example/jsons/suture_np/default_np.yaml --json ./ext/DreamerfD/example/jsons/suture_np/Dreamer.yaml --section 1 --logdir ./data/rl_log
  ```

### 3. Evaluation

  To evaluate a train model, you need to use ```--only-eval``` flag and ```--json``` for the directory to ```<train-log-path>/config.yaml```, for example:

  ```sh
  python ext/DreamerfD/example/suture_json.py --json ./data/rl_log/ambf-phantom-psm2-segment_script-zoom_needle_boximage-prefill8000-clutch6/DreamerfD/1/config.yaml --only-eval
  ```

### 4. Getting Familiar With `gym_np`

  you can play with `gym_np` interactively by typing
  ```sh
  source bash/ambf/init.sh
  python example/gym_env.py
  ```
  press keyboard `q` to quit and press other keys to proceed. For details, you can check out [example](./example/)

### 5. Reproducing Experiments.

  To reproduce experiments in [our paper](), you can refer to folder [exp](./exp)

### 6. Monitor Training with Tensorboard.

  To monitor the training process and results with tensorboard, type 

  ```sh
  source bash/ambf/init.sh
  tensorboard --logdir ./data/rl_log/
  ```




