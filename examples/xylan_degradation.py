# %% [markdown]
# # Xylan degradation
# 
# In this notebook we will explore xylan production in Bascillus subtilis 168

# %%
from spamdfba import toolkit as tk
import cobra
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import warnings
import rich
import multiprocessing as mp
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %% [markdown]
# First we have to load the metabolic model of Bascillus subtilis 168, which is available in the BiGG database. Run below to download the model.
# 

# %%
# !wget http://bigg.ucsd.edu/static/models/iYO844.json.gz
#! gzip -d iYO844.json.gz

# %%
from collections import Counter

# %% [markdown]
# Now let's load the model.

# %%
bacillus_model=cobra.io.load_json_model('./iYO844.json')

# %% [markdown]
# We have two genes known for xylan breakdown in Bascillus subtilis 168:
# 
# Endo-1,4-beta-xylanase A:P18429
# 
# Beta-xylosidase: P94489

# %%
P18429="MFKFKKNFLVGLSAALMSISLFSATASAASTDYWQNWTDGGGIVNAVNGSGGNYSVNWSNTGNFVVGKGWTTGSPFRTINYNAGVWAPNGNGYLTLYGWTRSPLIEYYVVDSWGTYRPTGTYKGTVKSDGGTYDIYTTTRYNAPSIDGDRTTFTQYWSVRQSKRPTGSNATITFSNHVNAWKSHGMNLGSNWAYQVMATEGYQSSGSSNVTVW"
P94489="MKITNPVLKGFNPDPSICRAGEDYYIAVSTFEWFPGVQIHHSKDLVNWHLVAHPLQRVSQLDMKGNPNSGGVWAPCLSYSDGKFWLIYTDVKVVDGAWKDCHNYLVTCETINGDWSEPIKLNSSGFDASLFHDTDGKKYLLNMLWDHRIDRHSFGGIVIQEYSDKEQKLIGKPKVIFEGTDRKLTEAPHLYHIGNYYYLLTAEGGTRYEHAATIARSANIEGPYEVHPDNPILTSWHDPGNPLQKCGHASIVQTHTDEWYLAHLTGRPIHPDDDSIFQQRGYCPLGRETAIQKLYWKDEWPYVVGGKEGSLEVDAPSIPETIFEATYPEVDEFEDSTLNINFQTLRIPFTNELGSLTQAPNHLRLFGHESLTSTFTQAFVARRWQSLHFEAETAVEFYPENFQQAAGLVNYYNTENWTALQVTHDEELGRILELTICDNFSFSQPLNNKIVIPREVKYVYLRVNIEKDKYYYFYSFNKEDWHKIDIALESKKLSDDYIRGGGFFTGAFVGMQCQDTSGNHIPADFRYFRYKEK"

# %%
def get_protein_production_reaction(protein_name:str,protein_sequence:str,atp_per_aa:float=4.2)->cobra.Reaction:
    aa_name_conversion = {
        "A": "ala__L_c",
        "R": "arg__L_c",
        "N": "asn__L_c",
        "D": "asp__L_c",
        "C": "cys__L_c",
        "Q": "gln__L_c",
        "E": "glu__L_c",
        "G": "gly_c",
        "H": "his__L_c",
        "I": "ile__L_c",
        "L": "leu__L_c",
        "K": "lys__L_c",
        "M": "met__L_c",
        "F": "phe__L_c",
        "P": "pro__L_c",
        "S": "ser__L_c",
        "T": "thr__L_c",
        "W": "trp__L_c",
        "Y": "tyr__L_c",
        "V": "val__L_c",
    }
    
    counts=dict([(aa_name_conversion[i],val) for i,val in Counter(protein_sequence).items()])
    counts["atp_c"]=len(protein_sequence)*atp_per_aa
    sum_=sum(counts.values())
    reaction=cobra.Reaction(protein_name+"_production")
    for i,val in counts.items():
        reaction.add_metabolites({cobra.Metabolite(i,compartment="c"):-val/sum_})
        
    reaction.add_metabolites({cobra.Metabolite(protein_name,compartment="e"):1/sum_})
    reaction.add_metabolites({cobra.Metabolite("adp_c",compartment="c"):counts["atp_c"]/sum_})
    reaction.add_metabolites({cobra.Metabolite("pi_c",compartment="c"):counts["atp_c"]/sum_})
    reaction.lower_bound=0
    reaction.upper_bound=100
    ex_reaction=cobra.Reaction(protein_name+"_export")
    ex_reaction.add_metabolites({reaction.products[0]:-1})
    return [reaction,ex_reaction]
    
    

# %%
get_protein_production_reaction("xylanase",P18429)
get_protein_production_reaction("xylosidase",P94489)

# %%
bacillus_model.add_reactions(get_protein_production_reaction("xylanase",P18429))
bacillus_model.add_reactions(get_protein_production_reaction("xylosidase",P94489))

# %%
bacillus_model.biomass_ind=bacillus_model.reactions.index("BIOMASS_BS_10")
agent1=tk.Agent("Bacllus_agent1",
                model=bacillus_model,
                actor_network=tk.ActorNN,
                critic_network=tk.CriticNN,
                clip=0.1,
                lr_actor=0.0005,
                lr_critic=0.001,
                grad_updates=5,
                action_ranges=[[-10,5],[-10,5]],
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
                observables=['Bacllus_agent1' ,"xyl__D_e", 'Xylan'],
                actions=["xylanase_production","xylosidase_production"],
                gamma=1,
                variance_handler=tk.sqrt_variance_handler,
                )

agents=[agent1]

# %%
ic={i[3:]:bacillus_model.medium[i] for i in bacillus_model.medium}

# %%
del ic["glc__D_e"]

# %%
constants=list(ic.keys())

# %%
def general_kinetics_xylanase(a,b):
    return 10*a*b/(0.5+a)  
def general_kinetics_xylosidase(a,b):
    return 30*a*b/(0.5+a)  

def variance_handler(batch_num):
    return max(0.5/np.sqrt(batch_num),0.01)

# %%
ic.update({"xyl__D_e":5,"Bacllus_agent1":0.01,"Xylan":1})
env_1=tk.Environment(name="Bacillus_168_Xylan",
                    agents=agents,
                    dilution_rate=0.00000001,
                    initial_condition=ic,
                    inlet_conditions={},
                    extracellular_reactions=[
                    {"reaction":{
                      "Xylose_oligo":100,
                      "Xylan":-1,},
                      "kinetics": (general_kinetics_xylanase,("Xylan","xylanase"))},                                                                  
                    {"reaction":{
                      "Xylose_oligo":-1,
                      "xyl__D_e":5,},
                      "kinetics": (general_kinetics_xylosidase,("Xylose_oligo","xylosidase"))},
                                           ],
                    constant=constants,
                     dt=0.5,
                     number_of_batches=10000,
                     episode_length=100,
                     episodes_per_batch=8,)

# %%
### train the agents actor network to output -0.5 and -0.5 (No xylanase and xylosidase production)

# %%
# initial_training_states=torch.rand(10000,4)*1000
# action_labels=torch.tensor([[-10.,-10.]]*10000)
# for i in range(1000):
#     outs=env_1.agents[0].actor_network_(torch.tensor(initial_training_states))
#     env_1.agents[0].optimizer_policy_.zero_grad()
#     loss=nn.MSELoss()(outs,action_labels)
#     loss.backward()
#     env_1.agents[0].optimizer_policy_.step()
#     if i%10==0:
#         print(loss)
    

sim_1=tk.Simulation(name=env_1.name,
                  env=env_1,
                  save_dir="./Results/",
                  save_every=10
                  )

# %%
env_1.agents[0].model.solver="gurobi"

# %%
sim_1.run(initial_critic_error=3000,parallel_framework="ray")

# %%

# tk.run_episode_single(env_1)

