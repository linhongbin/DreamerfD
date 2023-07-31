## Install gym_np benchmark

  - make sure install [DreamerfD](https://github.com/linhongbin/DreamerfD) 
  - modify the [setting.sh](bash/setting.sh) if your want to change settings.

  - Download gym_np:
  
    ```sh
    source ./bash/setting.sh
    mkdir -p $GYM_NP_PARRENT_DIR 
    cd $GYM_NP_PARRENT_DIR 
    git clone https://github.com/linhongbin/gym-np.git
    cd gym-np
    git submodule init
    git submodule update
    ```

  - Install gym_np:
  
    ```sh
    echo "ENV_NAME="dreamer_fd"" >> $GYM_NP_PARRENT_DIR/gym-np/bash/setting.sh
    source $GYM_NP_PARRENT_DIR/gym-np/bash/setting.sh # claim setting variables
    source $GYM_NP_PARRENT_DIR/gym-np/bash/install/ambf.sh # install AMBF, will take some time to compile
    source $GYM_NP_PARRENT_DIR/gym-np/bash/install/other.sh # install other packages
    ```
  

## Launch simulator

- **Launch roscore, open 1st terminal and type**
  ```sh
  roscore
  ```

- **Launch simulator, open 2st terminal and type**
  ```sh
  source ./bash/ambf/sim_standard.sh # For standard needle
  ```
  > For needle variations, type the following commands instead:
  >
  >  ```sh
  >  source $GYM_NP_PARRENT_DIR/gym-np/bash/ambf/sim_small.sh # >For small needle
  >  ```
  >  ```sh
  >  source $GYM_NP_PARRENT_DIR/gym-np/bash/ambf/sim_large.sh # For large needle
  >  ```
  >  ```sh
  >  source $GYM_NP_PARRENT_DIR/gym-np/bash/ambf/sim_irregular1.sh # For irregular shape 1
  >  ```
  >  ```sh
  >  source $GYM_NP_PARRENT_DIR/gym-np/bash/ambf/sim_irregular2.sh # For irregular shape 2
  >  ```

- **Launch crtk interface for control, open 3st terminal and type**
  ```sh
  source $GYM_NP_PARRENT_DIR/gym-np/bash/ambf/crtk.sh
  ```

- **Reset robotic arm to ready position, open 4st terminal and type**
  ```sh
  source $GYM_NP_PARRENT_DIR/gym-np/bash/ambf/reset.sh 
  ```
  ```
## Training

- **Train with DreamerfD**
  ```sh
  source $GYM_NP_PARRENT_DIR/gym-np/bash/init.sh
  python ./example/suture_json.py --section 1 -seed 1
  ```
  It will take 3 days on RTX3070 GPU for convergence of evaluated succeses rate (reaching 80%).
  > Resuming training is supported. To resume training, you need to use different seed number and use the same section number w.r.t. your preivous training, take the above command for example, you can run
  > ```sh
  > source $GYM_NP_PARRENT_DIR/gym-np/bash/init.sh
  > python ./example/suture_json.py --section 1 --seed 2
  >```


- **Train with other baselines**
  
  - Baseline 1: [Dreamer](https://danijar.com/project/dreamerv2/)
    ```sh
    source $GYM_NP_PARRENT_DIR/gym-np/bash/init.sh
    python ./example/suture_json.py --json ./example/jsons/suture_np/Dreamer.yaml --section 1 
    ```

## Evaluation

  To evaluate a train model, you need to use ```--only-eval``` flag and ```--json``` for the directory to ```<train-log-path>/config.yaml```, for example:

  ```sh
  source $GYM_NP_PARRENT_DIR/gym-np/bash/init.sh
  python ext/DreamerfD/example/suture_json.py --json ./data/rl_log/ambf-phantom-psm2-segment_script-zoom_needle_boximage-prefill8000-clutch6/DreamerfD/1/config.yaml --only-eval
  ```

### 4. Getting Familiar With `gym_np`

  you can play with `gym_np` interactively by typing
  ```sh
  source $GYM_NP_PARRENT_DIR/gym-np/bash/init.sh
  python example/gym_env.py
  ```
  press keyboard `q` to quit and press other keys to proceed. For details, you can check out [example](./example/)

### 5. Reproducing Experiments.

  To reproduce experiments in [our paper](), you can refer to folder [exp](./exp)

### 6. Monitor Training with Tensorboard.

  To monitor the training process and results with tensorboard, type 

  ```sh
  source $GYM_NP_PARRENT_DIR/gym-np/bash/init.sh
  tensorboard --logdir ./data/rl_log/
  ```




