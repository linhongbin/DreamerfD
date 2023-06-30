import gym
import surrol.gym
from gym_suture.env.surrol_wrapper import ImageObs
from efficient_dreamer import common
import efficient_dreamer.api_other as dv2
import argparse
import ruamel.yaml as yaml
import pathlib
parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default="")
parser.add_argument('--section', type=int, default=1) 
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--env', type=str, default="NeedlePick-v0")
parser.add_argument('--default-json', type=str, default="./example/jsons/surrol/default.yaml")
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
    baseline = "Efficient-Dreamer"

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
env.reset()
env = ImageObs(env)
dv2.train(env, config,time_limit=config.time_limit, bc_dir=args.bc)
