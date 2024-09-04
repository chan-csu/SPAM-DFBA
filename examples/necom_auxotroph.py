from spamdfba import toolkit as tk
from spamdfba import toymodels as tm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import multiprocessing as mp
import json
NUM_CORES = 8
warnings.filterwarnings("ignore")
agent1=tk.Agent("agent1",
				model=tm.Toy_Model_NE_Aux_1,
				actor_network=tk.ActorNN,
				critic_network=tk.CriticNN,
				clip=0.2,
				lr_actor=0.0001,
				lr_critic=0.001,
				grad_updates=4,
				action_ranges=[[-10,10],[-10,10]],
				optimizer_actor=torch.optim.Adam,
				optimizer_critic=torch.optim.Adam,       
				observables=['agent1','agent2','S',"A","B"],
				actions=['A_e','B_e'],
				gamma=1,
				)
agent2=tk.Agent("agent2",
				model=tm.Toy_Model_NE_Aux_2,
				actor_network=tk.ActorNN,
				critic_network=tk.CriticNN,
				clip=0.2,
				lr_actor=0.0001,
				lr_critic=0.001,
				grad_updates=4,
				action_ranges=[[-10,10],[-10,10]],
				optimizer_actor=torch.optim.Adam,
				optimizer_critic=torch.optim.Adam,       
				observables=['agent1','agent2','S',"A","B"],
				actions=['A_e','B_e'],
				gamma=1)
agents=[agent1,agent2]

env_aux=tk.Environment(name="Toy-NECOM-Auxotrophs-New",
 					agents=agents,
 					dilution_rate=0.0001,
 					extracellular_reactions=[],
 					initial_condition={"S":100,"agent1":0.1,"agent2":0.1},
 					inlet_conditions={"S":100},
 							dt=0.1,
 							number_of_batches=5000,
 							episodes_per_batch=int(NUM_CORES),)
sim_aux=tk.Simulation(name=env_aux.name,
                  env=env_aux,
                  save_dir="./Results/",
                  )
sim_aux.run(verbose=True,parallel_framework='ray')