from distutils.log import warn
import cobra
import torch
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal,Normal
import pickle
import time
import ray
import pandas as pd
import os
import time

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


def timer():
    "A simple classic decorator for timing a function/method"
    def timed(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Finished {func.__name__} in {end - start} seconds")
            return result
        return wrapper
    return timed




class Environment:
    """ An environment is a collection of the following:
        agents: a list of objects of class Agent
        extracellular reactions: a list of dictionaries. This list should look like this:
        {"reaction":{
            "a":1,
            "b":-1,
            "c":1
        },
        "kinetics": (lambda x,y: x*y,("a","b")),))}
     
    """
    def __init__(self,
                name:str,
                agents:list,
                extracellular_reactions:list[dict],
                initial_condition:dict,
                inlet_conditions:dict,
                batch_per_episode:int=1000,
                number_of_batches:int=100,
                dt:float=0.1,
                episode_time:float=1000,
                dilution_rate:float=0.05,
                max_c:dict={},
                episodes_per_batch:int=10,
                training:bool=True,
                constant:list=[]
                
                
                ) -> None:
        self.name=name
        self.agents = agents
        self.num_agents = len(agents)
        self.extracellular_reactions = extracellular_reactions
        self.dt = dt
        self.constant=constant
        self.episode_length = int(episode_time/dt)
        self.episodes_per_batch=episodes_per_batch
        self.number_of_batches=number_of_batches
        self.batch_per_episode = batch_per_episode
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
        self.min_c = np.zeros((len(self.species),))
        self.max_c = np.ones((len(self.species),))
        for key,value in max_c.items():
            self.max_c[self.species.index(key)]=value
        self.set_observables()
        self.set_networks()
        self.reset()
        
        print("Environment {} created successfully!.".format(self.name))

    
    def resolve_exchanges(self)->dict:
        """ Determines the exchange reaction mapping for the community."""
        models=[agent.model for agent in self.agents]
        return Build_Mapping_Matrix(models)
    
    def extract_species(self)->list:
        """ Determines the extracellular species in the community before extracellula reactions."""
        species=[ag.name for ag in self.agents]
        species.extend(self.mapping_matrix["Ex_sp"])
        return species

    def resolve_extracellular_reactions(self,extracellular_reactions:list[dict])->list[dict]:
        """ Determines the extracellular reactions for the community."""
        species=[]
        [species.extend(list(item["reaction"].keys())) for item in extracellular_reactions]
        new_species=[item for item in species if item not in self.species]
        if len(new_species)>0:
            warn("The following species are not in the community: {}".format(new_species))
            self.species.extend(list(set(new_species)))
        
    
    
    def reset(self):
        """ Resets the environment to its initial state."""
        self.state = self.initial_condition.copy()
    
    def step(self):
        """ Performs a single step in the environment."""
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
                
            Sols[i] = self.agents[i].model.optimize()
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



    def generate_random_c(self,size:int):
        """ Generates a random initial condition for the environment."""
        return np.random.uniform(low=self.min_c, high=self.max_c, size=(size,len(self.species))).T

    def batch_step(self,C:np.ndarray):
        """ Performs a batch of steps in the environment in parallel.
        This is just an experimental feature and is not yet implemented.
        C is a m*n where m is the number of species in the system and n 
        is the number of parallel steps.
        """
        batch_episodes=[]
        for batch in range(C.shape[1]):
            batch_episodes.append(Environment._step_p.remote(self,C[:,batch]))
        batch_episodes = ray.get(batch_episodes)

        return batch_episodes

    def set_observables(self):
        """ Sets the observables for the agents in the environment."""
        for agent in self.agents:
            agent.observables=[self.species.index(item) for item in agent.observables]

    def set_networks(self):
        """ Sets the networks for the agents in the environment."""
        if self.training==True:
            for agent in self.agents:
                agent.actor_network_=agent.actor_network(len(agent.observables)+1,len(agent.actions))
                agent.critic_network_=agent.critic_network(len(agent.observables)+1,1)
                agent.optimizer_value_ = agent.optimizer_critic(agent.critic_network_.parameters(), lr=agent.lr_critic)
                agent.optimizer_policy_ = agent.optimizer_actor(agent.actor_network_.parameters(), lr=agent.lr_actor)
    
class Agent:
    """ Any microbial agent will be an instance of this class.
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
                epsilon:float=0.01,
                lr_actor:float=0.001,
                lr_critic:float=0.001,
                buffer_sample_size:int=500,
                tau:float=0.001,
                alpha:float=0.001) -> None:

        self.name = name
        self.model = model
        self.optimizer_critic = optimizer_critic
        self.optimizer_actor = optimizer_actor
        self.gamma = gamma
        self.observables = observables
        self.actions = [self.model.reactions.index(item) for item in actions]
        self.observables = observables
        self.epsilon = epsilon
        self.general_uptake_kinetics=general_uptake
        self.tau = tau
        self.clip = clip
        self.actor_var = actor_var
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.buffer_sample_size = buffer_sample_size
        self.R=0
        self.grad_updates = grad_updates
        self.alpha = alpha
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.cov_var = torch.full(size=(len(self.actions),), fill_value=0.1)
        self.cov_mat = torch.diag(self.cov_var)
   
    def get_actions(self,observation:np.ndarray):
        """ 
        Gets the actions and their probabilities for the agent.
        """
        mean = self.actor_network_(torch.tensor(observation, dtype=torch.float32)).detach()
        # dist = MultivariateNormal(mean, self.actor_var)(mean, self.cov_mat)
        dist = Normal(mean, self.actor_var)
        action = dist.sample()
        log_prob =torch.sum(dist.log_prob(action))        # log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob
   
    def evaluate(self, batch_obs,batch_acts):
        V = self.critic_network_(batch_obs).squeeze()
        mean = self.actor_network_(batch_obs)
        # dist = MultivariateNormal(mean, self.cov_mat)
        dist = Normal(mean, self.actor_var)
        log_probs = torch.sum(dist.log_prob(batch_acts),dim=1)

        return V, log_probs 
    
    def compute_rtgs(self, batch_rews):

        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs


        
def Build_Mapping_Matrix(models:list[cobra.Model])->dict:
    """
    Given a list of COBRA model objects, this function will build a mapping matrix for all the exchange reactions.

    """

    Ex_sp = []
    Ex_rxns = []
    Temp_Map={}
    for model in models:
        Ex_rxns.extend([(model,list(model.reactions[rxn].metabolites)[0].id,rxn) for rxn in model.exchange_reactions if model.reactions[rxn].id.endswith("_e") and rxn!=model.biomass_ind])
    Ex_sp=list(set([item[1] for item in Ex_rxns]))
    Mapping_Matrix = np.full((len(Ex_sp), len(models)),-1, dtype=int)
    for record in Ex_rxns:
        Mapping_Matrix[Ex_sp.index(record[1]),models.index(record[0])]=record[2]

    return {"Ex_sp": Ex_sp, "Mapping_Matrix": Mapping_Matrix}

@ray.remote
def simulate(env,episodes=200,steps=1000):
    """ Simulates the environment for a given number of episodes and steps."""
    env.rewards=np.zeros((len(env.agents),episodes))
    env.record=[]
    for episode in range(episodes):
        env.reset()
        env.episode=episode

        for agent in env.agents:
            agent.rewards=[]
        C=[]
        episode_len=steps
        for ep in range(episode_len):
            env.t=episode_len-ep
            s,r,a,sp=env.step()
            for ind,ag in enumerate(env.agents):
                ag.rewards.append(r[ind])
                # ag.optimizer_reward_.zero_grad()
                # # r_pred=ag.reward_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), torch.FloatTensor(a[ind]))
                # # r_loss=nn.MSELoss()(r_pred,torch.FloatTensor(np.expand_dims(np.array(r[ind]),0)))
                # # r_loss.backward()
                # ag.optimizer_reward_.step()
                ag.optimizer_value_.zero_grad()
                Qvals = ag.critic_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), torch.FloatTensor(a[ind]))
                next_actions = ag.actor_network_(torch.FloatTensor(np.hstack([sp[ag.observables],env.t-1])))
                if env.t==1:
                    next_Q = torch.FloatTensor([0])
                else:
                    next_Q = ag.critic_network_.forward(torch.FloatTensor(np.hstack([sp[ag.observables],env.t-1])), next_actions.detach())
                Qprime = torch.FloatTensor(np.expand_dims(np.array(r[ind]),0))+ag.gamma*next_Q
                critic_loss=nn.MSELoss()(Qvals,Qprime.detach())
                critic_loss.backward()
                ag.optimizer_value_.step()
                ag.optimizer_policy_.zero_grad()
                policy_loss = -ag.critic_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])), ag.actor_network_(torch.FloatTensor(np.hstack([s[ag.observables],env.t])))).mean()
                policy_loss.backward()
                ag.optimizer_policy_.step()
            C.append(env.state.copy())
            env.record.append(np.hstack([env.state.copy(),np.reshape(np.array(env.temp_actions),(-1))]))

        # pd.DataFrame(C,columns=env.species).to_csv("Data.csv")

        for ag_ind,agent in enumerate(env.agents):
            print(episode)
            print(np.sum(agent.rewards))
            env.rewards[ag_ind,episode]=np.sum(agent.rewards)
    return env.rewards.copy(),env.record.copy()

def general_kinetic(x,y):
    return 0.1*x*y/(10+x)
def general_uptake(c):
    return 10*(c/(c+10))

def mass_transfer(x,y):
    return 0.01*(x-y)

def rollout(env):
    batch_obs={key.name:[] for key in env.agents}
    batch_acts={key.name:[] for key in env.agents}
    batch_log_probs={key.name:[] for key in env.agents}
    batch_rews = {key.name:[] for key in env.agents}
    batch_rtgs = {key.name:[] for key in env.agents}
    batch=[]
    for ep in range(env.episodes_per_batch):
        # batch.append(run_episode_single(env))
        batch.append(run_episode.remote(env))
    batch=ray.get(batch)
    for ep in range(env.episodes_per_batch):
        for ag in env.agents:
            batch_obs[ag.name].extend(batch[ep][0][ag.name])
            batch_acts[ag.name].extend(batch[ep][1][ag.name])
            batch_log_probs[ag.name].extend(batch[ep][2][ag.name])
            batch_rews[ag.name].append(batch[ep][3][ag.name])
    batch

    for ag in env.agents:
        env.rewards[ag.name].extend(list(np.sum(np.array(batch_rews[ag.name]),axis=1)))

    
    for agent in env.agents:

        batch_obs[agent.name] = torch.tensor(batch_obs[agent.name], dtype=torch.float)
        batch_acts[agent.name] = torch.tensor(batch_acts[agent.name], dtype=torch.float)
        batch_log_probs[agent.name] = torch.tensor(batch_log_probs[agent.name], dtype=torch.float)
        batch_rtgs[agent.name] = agent.compute_rtgs(batch_rews[agent.name]) 
    return batch_obs,batch_acts, batch_log_probs, batch_rtgs

@ray.remote
def run_episode(env):
    """ Runs a single episode of the environment. """
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
        start=time.time()      
        s,r,a,sp=env.step()
        end=time.time()
        for ind,ag in enumerate(env.agents):
            batch_obs[ag.name].append(np.hstack([s[ag.observables],env.t]))
            batch_acts[ag.name].append(a[ind])
            batch_log_probs[ag.name].append(ag.log_prob)
            episode_rews[ag.name].append(r[ind])
    return batch_obs,batch_acts, batch_log_probs, episode_rews

def run_episode_single(env):
    """ Runs a single episode of the environment. """
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
            agent.log_prob=log_prob .detach()        
        s,r,a,sp=env.step()
        for ind,ag in enumerate(env.agents):
            batch_obs[ag.name].append(np.hstack([s[ag.observables],env.t]))
            batch_acts[ag.name].append(a[ind])
            batch_log_probs[ag.name].append(ag.log_prob)
            episode_rews[ag.name].append(r[ind])
    return batch_obs,batch_acts, batch_log_probs, episode_rews


class Simulation:
    """This class is designed to run the final simulation for an environment and takes care of:
        - Saving the results given a specific interval
        - Plotting the results 
        This class can be extended later for added functionalities such as streaming the training results.
        
        Args:
    """
    
    def __init__(self,name:str,env:Environment,save_dir:str,save_every:int=200,overwrite:bool=False):
        self.name=name
        self.env=env
        self.save_dir=save_dir
        self.save_every=save_every
        self.overwrite=overwrite
        
    
    def run(self,solver:str="glpk",verbose:bool=True,initial_critic_error:float=100)->Environment:
        
        if not os.path.exists(self):
            os.makedirs(os.path.join(self.save_dir,self.env))
            
        for agent in self.env.agents:
            agent.model.solver="glpk"
            
        for batch in range(self.env.number_of_batches):
            batch_obs,batch_acts, batch_log_probs, batch_rtgs=rollout(self.env)  
            for agent in self.env.agents:
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
                        print("[bold green] Done![/bold green]")   
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
                        with open(os.path.join(self.save_dir,self.env.name,self.env.name+".pkl"), 'wb') as f:
                            pickle.dump(self.env, f)
                        with open(os.path.join(self.save_dir,self.env.name,self.env.name+"_obs.pkl"), 'wb') as f:
                            pickle.dump(batch_obs,f)
                        with open(os.path.join(self.save_dir,self.env.name,self.env.name+"_acts.pkl"), 'wb') as f:
                            pickle.dump(batch_acts,f)		
                    else:
                        with open(os.path.join(self.save_dir,self.env.name,self.env.name+f"_{batch}"+".pkl"), 'wb') as f:
                            pickle.dump(self.env, f)
                        with open(os.path.join(self.save_dir,self.env.name,self.env.name+f"_{batch}"+"_obs.pkl"), 'wb') as f:
                            pickle.dump(batch_obs,f)
                        with open(os.path.join(self.save_dir,self.env.name,self.env.name+f"_{batch}"+"_acts.pkl"), 'wb') as f:
                            pickle.dump(batch_acts,f)

                if verbose:
                    print(f"Batch {batch} finished:")
                    for agent in self.env.agents:
                        print(f"{agent.name} return was:  {np.mean(self.env.rewards[agent.name][-self.env.episodes_per_batch:])}")	