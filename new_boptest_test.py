import requests
import os
 
def create_log_dir(model_name,log_dir_name):
    models_dir = os.path.join("models",model_name)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)
    return models_dir

# url for the BOPTEST service
url = 'https://api.boptest.net'
 
# Select test case and get identifier
testcase = 'bestest_hydronic_heat_pump'
 
# Check if already started a test case and stop it if so before starting another
try:
  requests.put('{0}/stop/{1}'.format(url, testid))
except:
  pass
 
# Select and start a new test case
testid = \
requests.post('{0}/testcases/{1}/select'.format(url,testcase)).json()['testid']

# Get test case name
name = requests.get('{0}/name/{1}'.format(url, testid)).json()['payload']
print(name)

# Get test case name
name = requests.get('{0}/name/{1}'.format(url, testid)).json()['payload']
print(name)

from boptestGymEnv import (
    BoptestGymEnv,
    NormalizedObservationWrapper,
    DiscretizedActionWrapper,
)
from stable_baselines3 import DQN
import random
import numpy as np
import torch
import uuid
# Instantiate an RL agent with CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for random starting times of episodes
SEED = 123456
random.seed(SEED)
# Seed for random exploration and epsilon-greedy schedule
np.random.seed(SEED)
 
# Hyperparameters
LEARNING_RATE = 5e-4
BUFFER_SIZE = 365*24 # 1 year of buffer
LEARNING_STARTS = 24
TRAIN_FREQ = 1
BATCH_SIZE = 64
TIMESTEPS = 365*3 # 3 years of training per episode
#Format the log name to include the hyperparameters
 
 
LOG_NAME = 'DQN_lr_{}_buff_sz_{}_batch_sz_{}'.format(LEARNING_RATE,BUFFER_SIZE,BATCH_SIZE)
#Also include a UUID for the log name to avoid overwriting
LOG_NAME += '_'+str(uuid.uuid4())
#Create log directory and model name.
models_dir = create_log_dir(LOG_NAME,'logs')
# url for the BOPTEST service
url = "https://api.boptest.net"
# testcase = 'bestest_hydronic_heat_pump'
# Decide the state-action space of your test case
 
# Observations are time, zone temperature, outdoor temperature, solar radiation, internal gains, and electricity price
env = BoptestGymEnv(
        url                  = url,
        actions              = ['oveHeaPumY_u'],
        observations         = {'time':(0,604800),
                                'reaTZon_y':(280.,310.),
                                'TDryBul':(265,303),
                                'HDirNor':(0,862),
                                },
        random_start_time    = False,
        predictive_period=0,
        start_time           = 31*24*3600,
        max_episode_length   = 24*3600, # an episode is one day which corresponds to the timesteps
        warmup_period        = 24*3600,
        step_period          = 3600)
 
# Normalize observations and discretize action space
env = NormalizedObservationWrapper(env)
 
env = DiscretizedActionWrapper(env,n_bins_act=10)
 
# Instantiate an RL agent
model = DQN('MlpPolicy', env, verbose=1, gamma=0.99,device=device,
            learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, seed=SEED,
            buffer_size=BUFFER_SIZE, learning_starts=LEARNING_STARTS, train_freq=TRAIN_FREQ,tensorboard_log='logs',)
 
#Report time
import time
start_time = time.time()
# Main training loop
for i in range(10): # 30 years of training
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, tb_log_name=LOG_NAME,log_interval=1)
    model.save(f"{models_dir}/{TIMESTEPS*i}")
end_time = time.time()  
 
print(f"Training time: {end_time - start_time} seconds")
 
# Evaluate the trained agent
done = False
obs, _ = env.reset()
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs,reward,terminated,truncated,info = env.step(action)
    done = (terminated or truncated)
 
kpis=env.get_kpis()
print(kpis)
# Save the KPIs in a file
with open(f"{models_dir}/kpis.txt", "w") as file:
    file.write(str(kpis))
    file.close()
 
 