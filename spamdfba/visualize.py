import plotly.graph_objects as go
import plotly
import pickle
import pathlib
from typing import Iterable
import pandas as pd


COLORS=plotly.colors.DEFAULT_PLOTLY_COLORS
COLORS_TRANSPARENT=[f'rgba{color[3:-1]},0.2)' for color in COLORS]
class DataProcessor:
    def __init__(self, data):
        if not pathlib.Path(data).exists():
            raise FileNotFoundError(f"File {data} does not exist.")
        self.data = data
        
    
    def visualize_state(self,agent:str, on:Iterable[str]|None=None):
        with open(self.data, "rb") as f:
            data_ = pickle.load(f)
            batches_per_episode=data_['batches_per_episode']
            episode_length=data_['episode_length']
            states=data_["states"][agent]
            data_=data_["obs"][agent]
        data_=data_.numpy().reshape((-1,batches_per_episode,len(states)),order='F')
        fig = go.Figure()
        if on is None:
            on = states
        min_df = pd.DataFrame(data_.min(axis=1),columns=states)
        mean_df = pd.DataFrame(data_.mean(axis=1),columns=states)
        max_df = pd.DataFrame(data_.max(axis=1),columns=states)
        for index,state in enumerate(on):
            fig.add_trace(go.Scatter(x=list(range(len(data_))),y=mean_df[state],name=state,mode='lines',line=dict(color=COLORS[index])))
            fig.add_trace(go.Scatter(x=max_df.index+min_df.index[::-1],y=max_df[state]+min_df[state][::-1],fill='toself',fillcolor=COLORS_TRANSPARENT[index],line=dict(color='rgba(255,255,255,0)'),showlegend=False))
        fig.show()
    
    def visualize_action(self,agent:str, on:Iterable[str]|None=None):
        with open(self.data, "rb") as f:
            data_ = pickle.load(f)
            batches_per_episode=data_['batches_per_episode']
            episode_length=data_['episode_length']
            actions=data_["actions"][agent]
            data_=data_["acts"][agent]
        data_=data_.numpy().reshape((-1,batches_per_episode,len(actions)),order='F')
        fig = go.Figure()
        if on is None:
            on = actions
        min_df = pd.DataFrame(data_.min(axis=1),columns=actions)
        mean_df = pd.DataFrame(data_.mean(axis=1),columns=actions)
        max_df = pd.DataFrame(data_.max(axis=1),columns=actions)
        for index,action in enumerate(on):
            fig.add_trace(go.Scatter(x=list(range(len(data_))),y=mean_df[action],name=action,mode='lines',line=dict(color=COLORS[index])))
            fig.add_trace(go.Scatter(x=list(max_df.index)+list(min_df.index[::-1]),y=list(max_df[action])+list(min_df[action][::-1]),fill='toself',fillcolor=COLORS_TRANSPARENT[index],line=dict(color='rgba(255,255,255,0)'),showlegend=False))
        
        fig.show()
                                         
        
    
    
if __name__ == "__main__":
    dp = DataProcessor("/Users/parsaghadermarzi/Desktop/Academics/Projects/SPAM-DFBA/spamdfba/test_RL_perf/test_RL_perf_200.pkl")
    dp.visualize_action("agent1")