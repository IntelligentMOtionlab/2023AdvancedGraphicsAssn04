from stable_baselines3 import PPO
from custom_half_cheetah_v4 import CustomHalfCheetahEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env


def make_env(rank, obstacle=True, seed=0):
    def _init():
        env = CustomHalfCheetahEnv(obstacle = obstacle)
        env.reset(seed = seed + rank)
        return env
    return _init

## Argument Parser

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--obstacle', action='store_true', help='with obstacle')
## network path argument
parser.add_argument('--model_path', type=str, default="model", help='network path')

args = parser.parse_args()

if __name__ == "__main__":

    # Change the number of cpu you can use
    num_env = 4
    env = None
    
    env = SubprocVecEnv([make_env(i, args.obstacle) for i in range(num_env)]) 
    env = VecMonitor(env)

    # For Training 
    if not args.render:
        # Exploration noise
        policy_kwargs = dict(log_std_init = 0.0)
        model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./logs/")
        model.learn(total_timesteps=500000)
        model.save(args.model_path)
        
    # For rendering
    single_env = CustomHalfCheetahEnv(render_mode="human", obstacle = args.obstacle)
    model = PPO.load(args.model_path, env=single_env)
    single_env = model.get_env()
    obs = single_env.reset()
    
    # pos = single_env.env_method("get_x")[0] 

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = single_env.step(action)
        # if dones:
            # print(last_pos - pos)
            # pos = single_env.env_method("get_x")[0]
        last_pos = single_env.env_method("get_x")[0]
        single_env.render("human")
