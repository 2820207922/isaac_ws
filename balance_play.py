# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=True, enable_livestream=True)

# create task and register task
from balance_task import BalanceTask
task = BalanceTask(name="Balance")
env.set_task(task, backend="torch")

# import stable baselines
from stable_baselines3 import PPO

# Run inference on the trained policy
model = PPO.load("ppo_balance")
env._world.reset()
obs, _ = env.reset()
while env._simulation_app.is_running():
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action[0])
    print(f"action: {action}")
    # print(f"obs: {obs}, rewards: {rewards}, terminated: {terminated}, truncated: {truncated}, info: {info}")

env.close()