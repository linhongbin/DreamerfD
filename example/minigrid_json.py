import gym
import gym_minigrid
from dreamer_fd import common
import dreamer_fd.train as dv2
from gym.spaces import Dict
import argparse
import ruamel.yaml as yaml
import pathlib
parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default="")
parser.add_argument('--section', type=int, default=1) 
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--env', type=str, default="MiniGrid-DoorKey-6x6-v0")
parser.add_argument('--default-json', type=str, default="./example/jsons/minigrid/default_minigrid.yaml")
parser.add_argument('--bc', type=str, default="")

args = parser.parse_args()


configs = yaml.safe_load((pathlib.Path(args.default_json)).read_text())
config = common.Config(configs)
# print(config)
if args.json != "":
    configs = yaml.safe_load((pathlib.Path(args.json)).read_text())
    config = config.update(configs)
    baseline = pathlib.Path(args.json).stem
else:
    baseline = "DreamerBC-Plan"

if args.bc != "":
    baseline = "oracle-" + baseline
logdir = str(pathlib.Path('./data/minigrid') /args.env / baseline / str(args.section))
config = config.update({
'bc_dir': '',
'logdir': logdir,         
'seed': args.seed,
'jit': not args.debug,
                  })

env = gym.make(args.env)
env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
env.observation_space = Dict({k:v for k,v in env.observation_space.items() if k != 'mission'})
dv2.train(env, config,time_limit=config.time_limit, bc_dir=args.bc)
