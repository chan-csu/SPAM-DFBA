from spamdfba import toolkit as tk
from spamdfba import toymodels as tm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import multiprocessing as mp
import json
import pickle
NUM_CORES = 8
warnings.filterwarnings("ignore")
def slow_kinetics(x):
    return 10*x/(10+x)
def variance_handler(x):
    start=0.5
    end=0.01
    num_batches=10000
    slope=(start-end)/10000
    return start-x*slope

agent1=tk.Agent("agent1",
				model=tm.Toy_Model_NE_Aux_1,
				actor_network=tk.ActorNN,
				critic_network=tk.CriticNN,
				clip=0.1,
				lr_actor=0.0001,
				lr_critic=0.001,
				grad_updates=4,
				optimizer_actor=torch.optim.Adam,
				optimizer_critic=torch.optim.Adam,       
				observables=['agent1','agent2','S',"A","B"],
				actions=['A_e','B_e'],
				gamma=1,
				variance_handler=variance_handler,
				)
agent2=tk.Agent("agent2",
				model=tm.Toy_Model_NE_Aux_2,
				actor_network=tk.ActorNN,
				critic_network=tk.CriticNN,
				clip=0.1,
				lr_actor=0.0001,
				lr_critic=0.001,
				grad_updates=4,
				optimizer_actor=torch.optim.Adam,
				optimizer_critic=torch.optim.Adam,       
				observables=['agent1','agent2','S',"A","B"],
				actions=['A_e','B_e'],
				variance_handler=variance_handler,
				gamma=1)
agent1.general_uptake_kinetics=slow_kinetics
agent2.general_uptake_kinetics=slow_kinetics
agents=[agent1,agent2]

env_aux=tk.Environment(name="Toy-NECOM-Auxotrophs-oct-7-0-slow-kinetics",
 					agents=agents,
 					dilution_rate=0.0001,
 					extracellular_reactions=[],
 					initial_condition={"S":100,"agent1":0.1,"agent2":0.1},
 					inlet_conditions={"S":100},
 							dt=0.1,
 							number_of_batches=10000,
 							episodes_per_batch=8,)
sim_aux=tk.Simulation(name=env_aux.name,
                  env=env_aux,
                  save_dir="./Results/",
                  save_every=100		
                  )

sim_aux.run(verbose=True,parallel_framework="ray",pretrain_iter=10000)
 
# for i in [20,100]:
# 	with open('/Users/parsaghadermarzi/Desktop/Academics/Projects/SPAM-DFBA/examples/Results/Toy-NECOM-Auxotrophs-oct-7-0/Toy-NECOM-Auxotrophs-oct-7-0_9800_env.pkl', 'rb') as f:
# 		pretrained_env = pickle.load(f)
# 	pretrained_env.name = "Toy-NECOM-Auxotrophs-with-pretrained-"+str(i)
# 	pretrained_env.initial_condition[pretrained_env.species.index("A")] = i
# 	pretrained_env.initial_condition[pretrained_env.species.index("B")] = i
 
# 	sim_aux=tk.Simulation(name=pretrained_env.name,
# 	                  env=pretrained_env,
# 	                  save_dir="./Results/",
# 	                  )
# 	sim_aux.pre_trained=True
# 	sim_aux.run(verbose=True,parallel_framework='ray')