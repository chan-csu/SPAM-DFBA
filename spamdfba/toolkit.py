from distutils.log import warn
import cobra
import torch
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal,Normal
import pickle,logging
import time
import ray
import pandas as pd
import os
import time
import plotly.graph_objs as go
from rich.console import Console
from rich.table import Table
from typing import Iterable
import multiprocessing as mp
import warnings
import signal
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
# Ignore ray warnings to be communicated
warnings.filterwarnings("ignore")
ray.init(log_to_driver=False,ignore_reinit_error=True)
mp.set_start_method('fork')
mp.log_to_stderr()
DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']*10
DEFAULT_PLOTLY_COLORS_BACK=['rgba(31, 119, 180,0.2)', 'rgba(255, 127, 14,0.2)',
                       'rgba(44, 160, 44,0.2)', 'rgba(214, 39, 40,0.2)',
                       'rgba(148, 103, 189,0.2)', 'rgba(140, 86, 75,0.2)',
                       'rgba(227, 119, 194,0.2)', 'rgba(127, 127, 127,0.2)',
                       'rgba(188, 189, 34,0.2)', 'rgba(23, 190, 207,0.2)']*10


class NN(nn.Module):
    """
    This class is a subclass of nn.Module and is a general class for defining function approximators in the RL problems.
    
    Args:
        input_dim (int): dimension of the input, states, tensor.
        output_dim (int): dimension of the output tensor.
        hidden_dim (int): dimension of each hidden layer, defults to 20.
        activation : Type of the activation layer. Usually nn.Relu or nn.Tanh.
        n_hidden (int): number of hidden layers in the neural network.
     
    """
    def __init__(self,input_dim:int,output_dim:int,hidden_dim:int=20,activation=nn.ReLU,n_hidden:int=8)-> None:
        super(NN,self).__init__()
        self.inlayer=nn.Sequential(nn.Linear(input_dim,hidden_dim),activation())
        self.hidden=nn.Sequential(*[nn.Linear(hidden_dim,hidden_dim),activation()]*n_hidden)
        self.output=nn.Linear(hidden_dim,output_dim)
    
    def forward(self, obs:torch.FloatTensor)-> torch.FloatTensor:
        out=self.inlayer(obs)
        out=self.hidden(out)
        out=self.output(out)
        return out

class Agent:
    """ An object of this class defines an agent in the environment. At the core of an agent lies a COBRA model.
        Also the observable environment states are needed to be defined for an agent. Additionally, it should be defined what 
        reactions an agent have control over.
        

        Args:
            name (str): A descriptive name given to an agent.
            model (cobra.Model): A cobra model describing the metabolism of the agent.
            actor_network (NN): The neural network class, pyTorch, to be used for the actor network.
            critic_network (NN): The neural network class, pyTorch, to be used for the critic network.
            optimizer_critic (torch.optim.Adam): The Adam optimizer class used for tuning the critic network parameters.
            optimizer_actor (torch.optim.Adam): The Adam optimizer class used for tuning the actor network parameters.
            actions (list): list of reaction names that the agent has control over. The reactions should exist in the cobra model.
            observables (list): list of the names of metabolites that the agents can sense from the environment.
            clip (float): gradient clipping threshhold that is used in PPO algorithm
            actor_var (float): Amount of variance in the actor network suggestions. For exploration purpose. 
            grad_updates (int): How many steps of gradient decent is performed in each training step
            lr_actor (float) : The learning rate for the actor network 
            lr_critic (float) : The learning rate for the critic network 
        
        Examples:
            >>> from spamdfba import toymodels as tm
            >>> from spamdfba import toolkit as tk
            >>> agent1=tk.Agent("agent1",
                model=tm.ToyModel_SA.copy(),
                actor_network=tk.NN,
                critic_network=tk.NN,
                clip=0.1,
                lr_actor=0.0001,
                lr_critic=0.001,
                grad_updates=4,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
                observables=['agent1','agent2' ,'Glc', 'Starch'],
                actions=["Amylase_e"],
                gamma=1,
                )
    """
    def __init__(self,
                name:str,
                model:cobra.Model,
                actor_network:NN,
                critic_network:NN,
                optimizer_critic:torch.optim.Adam,
                optimizer_actor:torch.optim.Adam,
                actions:list[str],
                observables:list[str],
                gamma:float,
                clip:float=0.01,
                actor_var:float=0.1,
                grad_updates:int=1,
                lr_actor:float=0.001,
                lr_critic:float=0.001,
                ) -> None:

        self.name = name
        self.model = model
        self.optimizer_critic = optimizer_critic
        self.optimizer_actor = optimizer_actor
        self.gamma = gamma
        self.observables = observables
        self.actions = [self.model.reactions.index(item) for item in actions]
        self.observables = observables
        self.general_uptake_kinetics=general_uptake
        self.clip = clip
        self.actor_var = actor_var
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.grad_updates = grad_updates
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.cov_var = torch.full(size=(len(self.actions),), fill_value=0.1)
        self.cov_mat = torch.diag(self.cov_var)
   
    def get_actions(self,observation:np.ndarray):
        """ 
        This method will draw the actions from a normal distribution around the actor netwrok prediction.
        The derivatives are not calculated here.
        """
        mean = self.actor_network_(torch.tensor(observation, dtype=torch.float32)).detach()
        # dist = MultivariateNormal(mean, self.actor_var)(mean, self.cov_mat)
        dist = Normal(mean, self.actor_var)
        action = dist.sample()
        log_prob =torch.sum(dist.log_prob(action))        # log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob
   
    def evaluate(self, batch_obs:np.ndarray ,batch_acts:np.ndarray):
        """ 
        Calculates the value of the states, as well as the log probability af the actions that are taken.
        The derivatives are calculated here.
        """
        V = self.critic_network_(batch_obs).squeeze()
        mean = self.actor_network_(batch_obs)
        # dist = MultivariateNormal(mean, self.cov_mat)
        dist = Normal(mean, self.actor_var)
        log_probs = torch.sum(dist.log_prob(batch_acts),dim=1)

        return V, log_probs 
    
    def compute_rtgs(self, batch_rews:list):
        """Given a batch of rewards , it calculates the discouted return for each state for that batch"""

        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs



class Environment:
    """ An environment is a collection of the following:
        Agents: a list of objects of class Agent, defined below.
        extracellular reactions: a list of dictionaries that describes reaction that happens outside of cells.
        An example of such reactins is reactions catalyzed by extracellular enzymes. This list should look like this:
        [{"reaction":{
            "a":1,
            "b":-1,
            "c":1
        },
        "kinetics": (lambda x,y: x*y,("a","b")),))},...]
        
        Args:
            name (str): A descriptive name for the environment
            agents (Iterable): An iterable object like list or tuple including the collection of the agents to be used in the environment.
            extracellular_reactions (Iterable): An iterable object consisting of a collection of extracellular reactions defined as above.
            initial_condition (dict): A dictionary describing the initial concentration of all species in the environment to be used in the beginning
            of each state
            inlet_conditions (dict): A dictionary describing the inlet concentration of all species in the environment to be used in the beginning
            of each state. This is important when simulating continuous bioreactors as the concentration of the inlet stream should be taken into account.
            number_of_batches (int): Determines how many batches are performed in a simulation
            dt (float): Specifies the time step for DFBA calculations
            dilution_rate (float): The dilution rate of the bioreactor in per hour unit.
            episodes_per_batch (int): Determines how many episodes should be executed with same actor function in parallel (policy evaluation)
            episode_length (int): Determines how many time points exists within a given episode.
            training (bool): Whether to run in training mode. If false, no training happens.
            constant (list): A list of components that we want to hold their concentration constant during the simulations.
        
        Examples:
            >>> from spamdfba import toymodels as tm
            >>> from spamdfba import toolkit as tk
            >>> agent1=tk.Agent("agent1",
                model=tm.ToyModel_SA.copy(),
                actor_network=tk.NN,
                critic_network=tk.NN,
                clip=0.1,
                lr_actor=0.0001,
                lr_critic=0.001,
                grad_updates=4,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
                observables=['agent1','agent2' ,'Glc', 'Starch'],
                actions=["Amylase_e"],
                gamma=1,
                )
            >>> agent2=tk.Agent("agent2",
                model=tm.ToyModel_SA.copy(),
                actor_network=tk.NN,
                critic_network=tk.NN,
                clip=0.1,
                lr_actor=0.0001,
                lr_critic=0.001,
                grad_updates=4,
                optimizer_actor=torch.optim.Adam,
                optimizer_critic=torch.optim.Adam,
                observables=['agent1','agent2', 'Glc', 'Starch'],
                actions=["Amylase_e"],
                gamma=1,
                )
            >>> agents=[agent1,agent2]

            >>> env=tk.Environment(name="Toy-Exoenzyme-Two-agents",
                    agents=agents,
                    dilution_rate=0.0001,
                    initial_condition={"Glc":100,"agent1":0.1,"agent2":0.1,"Starch":10},
                    inlet_conditions={"Starch":10},
                    extracellular_reactions=[{"reaction":{
                    "Glc":10,
                    "Starch":-0.1,},
                    "kinetics": (tk.general_kinetic,("Glc","Amylase"))}],
                           dt=0.1,
                           number_of_batches=1000,
                           episodes_per_batch=int(NUM_CORES/2),
                           )       
    """
    def __init__(self,
                name:str,
                agents:Iterable,
                extracellular_reactions:Iterable[dict],
                initial_condition:dict,
                inlet_conditions:dict,
                number_of_batches:int=100,
                dt:float=0.1,
                dilution_rate:float=0.05,
                episodes_per_batch:int=10,
                episode_length:int=1000,
                training:bool=True,
                constant:list=[]
                
                
                ) -> None:
        self.name=name
        self.agents = agents
        self.num_agents = len(agents)
        self.extracellular_reactions = extracellular_reactions
        self.dt = dt
        self.constant=constant
        self.episodes_per_batch=episodes_per_batch
        self.number_of_batches=number_of_batches
        self.dilution_rate = dilution_rate
        self.training=training
        self.mapping_matrix=self.resolve_exchanges()
        self.species=self.extract_species()
        self.resolve_extracellular_reactions(extracellular_reactions)
        self.initial_condition =np.zeros((len(self.species),))
        for key,value in initial_condition.items():
            self.initial_condition[self.species.index(key)]=value
        self.inlet_conditions = np.zeros((len(self.species),))
        for key,value in inlet_conditions.items():
            self.inlet_conditions[self.species.index(key)]=value
        self.set_observables()
        self.set_networks()
        self.reset()
        self.time_dict={
            "optimization":[],
            "step":[],
            "episode":[]
                        }
        self.episode_length=episode_length
        self.rewards={agent.name:[] for agent in self.agents}
        
        

    
    def resolve_exchanges(self)->dict:
        """ Determines the exchange reaction mapping for the community. This mapping is required to keep track of 
        Metabolite pool change by relating exctacellular concentrations with production or consumption by the agents."""
        models=[agent.model for agent in self.agents]
        return Build_Mapping_Matrix(models)
    
    def extract_species(self)->list:
        """ Determines the extracellular species in the community before extracellula reactions."""
        species=[ag.name for ag in self.agents]
        species.extend(self.mapping_matrix["Ex_sp"])
        return species

    def resolve_extracellular_reactions(self,extracellular_reactions:list[dict])->None:
        """ Determines the extracellular reactions for the community. This method adds any new compounds required to run DFBA
        to the system.
        Args:
            extracellular_reactions list[dict]: list of extracellular reactions as defined in the constructor.
        """
        species=[]
        [species.extend(list(item["reaction"].keys())) for item in extracellular_reactions]
        new_species=[item for item in species if item not in self.species]
        if len(new_species)>0:
            warn("The following species are not in the community: {}".format(new_species))
            self.species.extend(list(set(new_species)))
        
    
    
    def reset(self):
        """ Resets the environment to its initial state."""
        self.state = self.initial_condition.copy()
        self.rewards={agent.name:[] for agent in self.agents}
        self.time_dict={
            "optimization":[],
            "step":[],
            "episode":[]
                        }
    
    def step(self)-> tuple[np.ndarray,list,list,np.ndarray]:
        """ Performs a single DFBA step in the environment.
        This method provides similar interface as other RL libraries: It returns:
        current state, rewards given to each agent from FBA calculations, actions each agent took,
        and next state calculated similar to DFBA.
        """
        self.temp_actions=[]
        self.state[self.state<0]=0
        dCdt = np.zeros(self.state.shape)
        Sols = list([0 for i in range(len(self.agents))])
        for i,M in enumerate(self.agents):
            for index,item in enumerate(self.mapping_matrix["Ex_sp"]):
                if self.mapping_matrix['Mapping_Matrix'][index,i]!=-1:
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].upper_bound=100
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].lower_bound=-M.general_uptake_kinetics(self.state[index+len(self.agents)])


            for index,flux in enumerate(M.actions):
                if M.a[index]<0:
                    M.model.reactions[M.actions[index]].lower_bound=max(M.a[index],M.model.reactions[M.actions[index]].lower_bound)
                else:
                    M.model.reactions[M.actions[index]].lower_bound=min(M.a[index],10)
            t_0=time.time() 
            Sols[i] = self.agents[i].model.optimize()
            self.time_dict["optimization"].append(time.time()-t_0)
            if Sols[i].status == 'infeasible':
                self.agents[i].reward=-1
                dCdt[i] = 0
            else:
                dCdt[i] += Sols[i].objective_value*self.state[i]
                self.agents[i].reward =Sols[i].objective_value*self.state[i]

        for i in range(self.mapping_matrix["Mapping_Matrix"].shape[0]):
        
            for j in range(len(self.agents)):

                if self.mapping_matrix["Mapping_Matrix"][i, j] != -1:
                    if Sols[j].status == 'infeasible':
                        dCdt[i+len(self.agents)] += 0
                    else:
                        dCdt[i+len(self.agents)] += Sols[j].fluxes.iloc[self.mapping_matrix["Mapping_Matrix"]
                                                    [i, j]]*self.state[j]
        
        # Handling extracellular reactions

        for ex_reaction in self.extracellular_reactions:
            rate=ex_reaction["kinetics"][0](*[self.state[self.species.index(item)] for item in ex_reaction["kinetics"][1]])
            for metabolite in ex_reaction["reaction"].keys():
                dCdt[self.species.index(metabolite)]+=ex_reaction["reaction"][metabolite]*rate
        dCdt+=self.dilution_rate*(self.inlet_conditions-self.state)
        C=self.state.copy()
        for item in self.constant:
            dCdt[self.species.index(item)]=0
        self.state += dCdt*self.dt

        Cp=self.state.copy()
        return C,list(i.reward for i in self.agents),list(i.a for i in self.agents),Cp


    def set_observables(self):
        """ Sets the observables for the agents in the environment."""
        for agent in self.agents:
            agent.observables=[self.species.index(item) for item in agent.observables]

    def set_networks(self):
        """ Sets up the networks and optimizers for the agents in the environment."""
        if self.training==True:
            for agent in self.agents:
                agent.actor_network_=agent.actor_network(len(agent.observables)+1,len(agent.actions))
                agent.critic_network_=agent.critic_network(len(agent.observables)+1,1)
                agent.optimizer_value_ = agent.optimizer_critic(agent.critic_network_.parameters(), lr=agent.lr_critic)
                agent.optimizer_policy_ = agent.optimizer_actor(agent.actor_network_.parameters(), lr=agent.lr_actor)
    
    def copy(self):
        """ Returns a deep copy of the environment."""
        return copy.deepcopy(self)
        
    

        
def Build_Mapping_Matrix(models:list[cobra.Model])->dict:
    """
    Given a list of COBRA model objects, this function will build a mapping matrix for all the exchange reactions.

    """

    Ex_sp = []
    Ex_rxns = []
    for model in models:
        Ex_rxns.extend([(model,list(model.reactions.get_by_id(rxn.id).metabolites)[0].id,model.reactions.index(rxn.id)) for rxn in model.exchanges if model.reactions.index(rxn)!=model.biomass_ind])
    Ex_sp=list(set([item[1] for item in Ex_rxns]))
    Mapping_Matrix = np.full((len(Ex_sp), len(models)),-1, dtype=int)
    for record in Ex_rxns:
        Mapping_Matrix[Ex_sp.index(record[1]),models.index(record[0])]=record[2]

    return {"Ex_sp": Ex_sp, "Mapping_Matrix": Mapping_Matrix}



def general_kinetic(x:float,y:float)->float:
    """A simple function implementing MM kinetics """
    return 0.1*x*y/(10+x)
def general_uptake(c:float)->float:
    """An extremely simple function for mass transfer kinetic. Only used for testing """
    return 10*(c/(c+10))

def mass_transfer(x:float,y:float,k:float=0.01)->float:
    """A simple function for mass transfer kinetic """
    return k*(x-y)

def rollout(env:Environment,num_workers:int|None=None,parallel_framework:str="native")->tuple:
    """Performs a batch calculation in parallel using Ray library.
    Args:
        env (Environment): The environment instance to run the episodes for
        num_workers (int): Number of workers to be used in parallel computation. If None, the number of workers is equal to the number of episodes per batch.
        parallel_framework (str): The parallel framework to be used. Currently you can choose between "ray" and "native". The native framework is a simple parallelization using python's multiprocessing library.
    """
    if num_workers is None:
        num_workers=env.episodes_per_batch
    t0_batch=time.time()
    batch_obs={key.name:[] for key in env.agents}
    batch_acts={key.name:[] for key in env.agents}
    batch_log_probs={key.name:[] for key in env.agents}
    batch_rews = {key.name:[] for key in env.agents}
    batch_rtgs = {key.name:[] for key in env.agents}
    batch_times={"step":[],
                 "episode":[],
                 "optimization":[],
                 "batch":[]}

    batch=[]
    env.reset()
    if parallel_framework=="ray":
        for ep in range(num_workers):
            # batch.append(run_episode_single(env))
            batch.append(run_episode_ray.remote(env))
        batch=ray.get(batch)
    
    elif parallel_framework=="native":
        with mp.Pool(num_workers,initializer=_initialize_mp()) as pool:
            batch=[pool.apply_async(run_episode_single,args=(env,)) for i in range(num_workers)]
            batch=[item.get() for item in batch]
            
    else:
        raise ValueError("The parallel framework should be either 'ray' or 'native'")
    for ep in range(num_workers):
        for ag in env.agents:
            batch_obs[ag.name].extend(batch[ep][0][ag.name])
            batch_acts[ag.name].extend(batch[ep][1][ag.name])
            batch_log_probs[ag.name].extend(batch[ep][2][ag.name])
            batch_rews[ag.name].append(batch[ep][3][ag.name])
        batch_times["step"].extend(batch[ep][4]["step"])
        batch_times["episode"].extend(batch[ep][4]["episode"])
        batch_times["optimization"].extend(batch[ep][4]["optimization"])
    
    for ag in env.agents:
        env.rewards[ag.name].extend(list(np.sum(np.array(batch_rews[ag.name]),axis=1)))
        
    for agent in env.agents:

        batch_obs[agent.name] = torch.tensor(batch_obs[agent.name], dtype=torch.float)
        batch_acts[agent.name] = torch.tensor(batch_acts[agent.name], dtype=torch.float)
        batch_log_probs[agent.name] = torch.tensor(batch_log_probs[agent.name], dtype=torch.float)
        batch_rtgs[agent.name] = agent.compute_rtgs(batch_rews[agent.name]) 
    batch_times["batch"].append(time.time()-t0_batch)
    return batch_obs,batch_acts, batch_log_probs, batch_rtgs,batch_times,env.rewards.copy()

@ray.remote
def run_episode_ray(env:Environment)->tuple:
    """ Runs a single episode of the environment used for parallel computatuon of episodes.
    """
    t_0_ep=time.time()
    batch_obs = {key.name:[] for key in env.agents}
    batch_acts = {key.name:[] for key in env.agents}
    batch_log_probs = {key.name:[] for key in env.agents}
    episode_rews = {key.name:[] for key in env.agents}
    env.reset()
    episode_len=env.episode_length
    for ep in range(episode_len):
        env.t=episode_len-ep
        obs = env.state.copy()
        for agent in env.agents:   
            action, log_prob = agent.get_actions(np.hstack([obs[agent.observables],env.t]))
            agent.a=action
            agent.log_prob=log_prob.detach() 
        t_0_step=time.time()
        s,r,a,sp=env.step()
        env.time_dict["step"].append(time.time()-t_0_step)
        for ind,ag in enumerate(env.agents):
            batch_obs[ag.name].append(np.hstack([s[ag.observables],env.t]))
            batch_acts[ag.name].append(a[ind])
            batch_log_probs[ag.name].append(ag.log_prob)
            episode_rews[ag.name].append(r[ind])
        env.time_dict["step"].append(time.time()-t_0_step)
    env.time_dict["episode"].append(time.time()-t_0_ep)
    return batch_obs,batch_acts, batch_log_probs, episode_rews,env.time_dict,env.rewards
    

def run_episode_single(env):
    """ Runs a single episode of the environment used for parallel computatuon of episodes.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    t_0_ep=time.time()
    batch_obs = {key.name:[] for key in env.agents}
    batch_acts = {key.name:[] for key in env.agents}
    batch_log_probs = {key.name:[] for key in env.agents}
    episode_rews = {key.name:[] for key in env.agents}
    env.reset()
    episode_len=env.episode_length
    for ep in range(episode_len):
        env.t=episode_len-ep
        obs = env.state.copy()
        for agent in env.agents:   
            action, log_prob = agent.get_actions(np.hstack([obs[agent.observables],env.t]))
            agent.a=action
            agent.log_prob=log_prob.detach() 
        t_0_step=time.time()
        s,r,a,sp=env.step()
        env.time_dict["step"].append(time.time()-t_0_step)
        for ind,ag in enumerate(env.agents):
            batch_obs[ag.name].append(np.hstack([s[ag.observables],env.t]))
            batch_acts[ag.name].append(a[ind])
            batch_log_probs[ag.name].append(ag.log_prob)
            episode_rews[ag.name].append(r[ind])
        env.time_dict["step"].append(time.time()-t_0_step)
    env.time_dict["episode"].append(time.time()-t_0_ep)
    return batch_obs,batch_acts, batch_log_probs, episode_rews,env.time_dict,env.rewards


class Simulation:
    """This class is designed to run the final simulation for an environment and additionaly does:
        - Saving the results given a specific interval
        - Plotting the results 
        - calculating the duration of different parts of the code.
        This class can be extended easily later for added functionalities such as online streaming the training results.
        
        Args:
            name (str): A descriptive name given to the simulation. This name is used to save the training files.
            env (environment): The environment to perform the simulations in. 
            save_dir (str): The DIRECTORY to which you want to save the training results
            overwrite (bool): Determines whether to overwrite the pickel in each saving interval create new files
            report (dict): Includes the reported time at each step
    """
    
    def __init__(self,name:str,env:Environment,save_dir:str,save_every:int=200,overwrite:bool=False):
        self.name=name
        self.env=env
        self.save_dir=save_dir
        self.save_every=save_every
        self.overwrite=overwrite
        self.report={}
        
    
    def run(self,solver:str="glpk",verbose:bool=True,initial_critic_error:float=100,parallel_framework:str="native")->Environment:
        """This method runs the training loop
        
        Args:
            solver (str): The solver to be used by cobrapy
            verbose (bool): whether to print the training results after each iteration
            initial_critic_error (float): To make the training faster this method first trains the critic network on the first batch of episodes to
            make the critic network produce more realistic values in the beginning. This parameter defines what is the allowable MSE of the critic network
            on the first batch of data obtained from the evironment
        Returns:
            Environment: The trained version of the environment.
            
            """
        t_0_sim=time.time()
        self.report={"returns":{ag.name:[] for ag in self.env.agents}}
        self.report["times"]={
            "step":[],
            "optimization":[],
            "batch":[],
            "simulation":[]
        }            
        if not os.path.exists(os.path.join(self.save_dir,self.name)):
            os.makedirs(os.path.join(self.save_dir,self.name))
            
        for agent in self.env.agents:
            agent.model.solver=solver
            
        for batch in range(self.env.number_of_batches):
            batch_obs,batch_acts, batch_log_probs, batch_rtgs,batch_times,env_rew=rollout(self.env,parallel_framework=parallel_framework) 
            self.report["times"]["step"].append(np.mean(batch_times["step"]))
            self.report["times"]["optimization"].append(np.mean(batch_times["optimization"]))
            self.report["times"]["batch"].append(np.mean(batch_times["batch"]))
            for agent in self.env.agents:
                self.report["returns"][agent.name].append(env_rew[agent.name])
                V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])  
                A_k = batch_rtgs[agent.name] - V.detach()       
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-5)   
                if batch==0:
                    if verbose:
                        print("Hold on, bringing the creitc network to range ...")
                        err=initial_critic_error+1
                        while err>initial_critic_error:   
                            V, _= agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])  
                            critic_loss = nn.MSELoss()(V, batch_rtgs[agent.name])   
                            agent.optimizer_value_.zero_grad()  
                            critic_loss.backward()  
                            agent.optimizer_value_.step()   
                            err=critic_loss.item()  
                    if verbose:
                        print("Done!")   
                else: 
                    for _ in range(agent.grad_updates):                                                      
                        
                        V, curr_log_probs = agent.evaluate(batch_obs[agent.name],batch_acts[agent.name])
                        ratios = torch.exp(curr_log_probs - batch_log_probs[agent.name])
                        surr1 = ratios * A_k.detach()
                        surr2 = torch.clamp(ratios, 1 - agent.clip, 1 + agent.clip) * A_k
                        actor_loss = (-torch.min(surr1, surr2)).mean()
                        critic_loss = nn.MSELoss()(V, batch_rtgs[agent.name])
                        agent.optimizer_policy_.zero_grad()
                        actor_loss.backward(retain_graph=False)
                        agent.optimizer_policy_.step()
                        agent.optimizer_value_.zero_grad()
                        critic_loss.backward()
                        agent.optimizer_value_.step()                                                            

                if batch%self.save_every==0:
                    if self.overwrite:
                        with open(os.path.join(self.save_dir,self.name,self.name+".pkl"), 'wb') as f:
                            pickle.dump(self.env, f)
                        with open(os.path.join(self.save_dir,self.name,self.name+"_obs.pkl"), 'wb') as f:
                            pickle.dump(batch_obs,f)
                        with open(os.path.join(self.save_dir,self.name,self.name+"_acts.pkl"), 'wb') as f:
                            pickle.dump(batch_acts,f)		
                    else:
                        with open(os.path.join(self.save_dir,self.name,self.name+f"_{batch}"+".pkl"), 'wb') as f:
                            pickle.dump(self.env, f)
                        with open(os.path.join(self.save_dir,self.name,self.name+f"_{batch}"+"_obs.pkl"), 'wb') as f:
                            pickle.dump(batch_obs,f)
                        with open(os.path.join(self.save_dir,self.name,self.name+f"_{batch}"+"_acts.pkl"), 'wb') as f:
                            pickle.dump(batch_acts,f)

            if verbose:
                print(f"Batch {batch} finished:")
                for agent in self.env.agents:
                    print(f"{agent.name} return was:  {np.mean(self.env.rewards[agent.name][-self.env.episodes_per_batch:])}")

        self.report["times"]["simulation"].append(time.time()-t_0_sim)
        
    def plot_learning_curves(self,plot:bool=True)->go.Figure:
        """
        This method plots the learning curve for all the agents.
        Args:
            plot (bool): whether to render the plot as well 
        
        Returns:
            go.Figure : Returns a plotly figure for learning curves of the agents.
        """
        fig = go.Figure()
        for index,agent in enumerate(self.env.agents):
            rets=pd.DataFrame(self.report["returns"][agent.name])
            x=rets.index.to_list()
            fig.add_trace(go.Scatter(
                x=x,
                y=rets.mean(axis=1).to_list(),
                line=dict(color=DEFAULT_PLOTLY_COLORS[index]),
                name=agent.name,
                mode='lines'
                        ))
            fig.add_trace(go.Scatter(
                        x=x+x[::-1],
                        y=rets.max(axis=1).to_list()+rets.min(axis=1).to_list()[::-1],
                        fill='toself',
                        fillcolor=DEFAULT_PLOTLY_COLORS_BACK[index],
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False)
                            )
            fig.update_layout(
                xaxis={
                    "title":"Batch"
                },
                yaxis={
                    "title":"Total Episode Return"
                }
                
            )
        if plot:
            fig.show()
        return fig
            
    def print_training_times(self,draw_table:bool=True)->list[dict]:
        """Returns a dictionary describing the simulation time at different level of the training process. You can also opt to draw a table based on this results 
        using Rich library.
        
        Args:
            draw_table (bool): whether to draw the table in the console
            
        Returns:
            dict: A list of dictionaries that contain duration of execution for different stages of simulation 
        """
        report_times=pd.concat([pd.DataFrame.from_dict(self.report["times"],orient='index').fillna(method="ffill",axis=1).mean(axis=1),pd.DataFrame.from_dict(self.report["times"],orient='index').fillna(method="ffill",axis=1).std(axis=1)],axis=1).rename({0:"mean",1:"std"},axis=1).to_dict(orient="record")
        if draw_table:
            table = Table(title="Simulation times")
            table.add_column("Level", justify="left", style="cyan", no_wrap=True)
            table.add_column("Mean(s)", style="cyan",justify="left")
            table.add_column("STD(s)", justify="left", style="cyan")
            table.add_row("Optimization",str(report_times[1]["mean"]),str(report_times[1]["std"]))
            table.add_row("Step",str(report_times[0]["mean"]),str(report_times[0]["std"]))
            table.add_row("Batch",str(report_times[2]["mean"]),str(report_times[2]["std"]))
            table.add_row("Simulation",str(report_times[3]["mean"]),"NA")
            console = Console()
            console.print(table)
        return report_times


def replicate_env(env:Environment,num)->list[Environment]:
    envs=[]
    s=pickle.dumps(env)
    for i in range(num):
        envs.append(pickle.loads(s))
    return envs

def _initialize_mp():
    torch.seed()
    np.random.seed()
    
    
if __name__=="__main__":
    pass