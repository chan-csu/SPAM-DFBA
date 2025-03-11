import plotly.graph_objects as go
import plotly
import pickle
import pathlib
from typing import Iterable
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import numpy as np


COLORS = px.colors.qualitative.Plotly+px.colors.qualitative.Dark24
COLORS_TRANSPARENT=[f'rgba{pc.hex_to_rgb(color) + (0.2,)}' for color in COLORS]
class DataProcessor:
    def __init__(self, data):
        if not pathlib.Path(data).exists():
            raise FileNotFoundError(f"File {data} does not exist.")
        with open(data,"rb") as f:
            self.data = pickle.load(f)
        
    
    def visualize_state(self,agent:str, on:Iterable[str]|None=None):
        data_ = self.data
        batches_per_episode = data_['batches_per_episode']
        episode_length = data_['episode_length']
        states = data_["states"][agent]
        data_ = data_["obs"][agent]
        data_ = data_.numpy().reshape((-1, batches_per_episode, len(states)), order='F')
        fig = go.Figure()
        if on is None:
            on = states
        min_df = pd.DataFrame(data_.min(axis=1), columns=states)
        mean_df = pd.DataFrame(data_.mean(axis=1), columns=states)
        max_df = pd.DataFrame(data_.max(axis=1), columns=states)
        for index, state in enumerate(on):
            fig.add_trace(go.Scatter(x=list(range(len(data_))), y=mean_df[state], name=state, mode='lines', line=dict(color=COLORS[index])))
            fig.add_trace(go.Scatter(x=max_df.index+min_df.index[::-1], y=max_df[state]+min_df[state][::-1], fill='toself', fillcolor=COLORS_TRANSPARENT[index], line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        return fig
    
    def visualize_action(self,agent:str, on:Iterable[str]|None=None):
        data_ = self.data
        batches_per_episode = data_['batches_per_episode']
        episode_length = data_['episode_length']
        actions = data_["actions"][agent]
        data_ = data_["acts"][agent]
        data_ = data_.numpy().reshape((-1, batches_per_episode, len(actions)), order='F')
        fig = go.Figure()
        if on is None:
            on = actions
        min_df = pd.DataFrame(data_.min(axis=1), columns=actions)
        mean_df = pd.DataFrame(data_.mean(axis=1), columns=actions)
        max_df = pd.DataFrame(data_.max(axis=1), columns=actions)
        for index, action in enumerate(on):
            fig.add_trace(go.Scatter(x=list(range(len(data_))), y=mean_df[action], name=action, mode='lines', line=dict(color=COLORS[index])))
            fig.add_trace(go.Scatter(x=list(max_df.index)+list(min_df.index[::-1]), y=list(max_df[action])+list(min_df[action][::-1]), fill='toself', fillcolor=COLORS_TRANSPARENT[index], line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        
        return fig
    
    def visualize_reward(self,agent:str):
        data_ = self.data
        batches_per_episode = data_['batches_per_episode']
        episode_length = data_['episode_length']
        rews =np.array(data_["rews"][agent]).reshape((-1,batches_per_episode),order='F').mean(axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(rews))), y=rews, name="Reward", mode='lines', line=dict(color=COLORS[0])))
        return fig
                                         
                                         
class CompareDataProcessor:
    def __init__(self, data:list[str]):
        self.data=data
    
    def compare_state(self, agent:str, on:str="all"):
        collector = {key: [
            pd.DataFrame(value.data["obs"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["states"][agent])), order='F').mean(axis=1), columns=value.data["states"][agent]),
            pd.DataFrame(value.data["obs"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["states"][agent])), order='F').min(axis=1), columns=value.data["states"][agent]),
            pd.DataFrame(value.data["obs"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["states"][agent])), order='F').max(axis=1), columns=value.data["states"][agent])
        ] for key, value in self.data.items()}
        
        fig = go.Figure()
        
        if on == 'all':
            for key, color in zip(collector.keys(), COLORS, strict=False):
                for state in collector[key][0].columns:
                    fig.add_trace(go.Scatter(x=collector[key][0].index, y=collector[key][0][state], mode='lines', line={"color": color}, legendgroup=str(key)+str(state), name=str(key)+"_"+str(state)))
                    fig.add_trace(go.Scatter(x=collector[key][1].index.tolist() + collector[key][2].index[::-1].tolist(),
                                             y=collector[key][1][state].tolist() + collector[key][2][state][::-1].tolist(),
                                             fill='toself',
                                             fillcolor='rgba' + str(pc.hex_to_rgb(color) + (0.2,)),
                                             name=key,
                                             legendgroup=str(key)+str(state),
                                             showlegend=False,
                                             line=dict(color='rgba' + str(pc.hex_to_rgb(color) + (0,)))))
        else:
            for index, key in enumerate(collector.keys()):
                fig.add_trace(go.Scatter(x=collector[key][0].index, y=collector[key][0][on], mode='lines', legendgroup=key, name=key, line={"color": COLORS[index]}))
                fig.add_trace(go.Scatter(x=collector[key][1].index.tolist() + collector[key][2].index[::-1].tolist(),
                                         y=collector[key][1][on].tolist() + collector[key][2][on][::-1].tolist(),
                                         fill='toself',
                                         fillcolor='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0.2,)),
                                         name=key,
                                         legendgroup=key,
                                         showlegend=False,
                                         line=dict(color='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0,)))))

        fig.show()
        
    
    
    def compare_action(self, agent:str, on:str="all"):
        collector = {key: [
            pd.DataFrame(value.data["acts"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["actions"][agent])), order='F').mean(axis=1), columns=value.data["actions"][agent]),
            pd.DataFrame(value.data["acts"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["actions"][agent])), order='F').min(axis=1), columns=value.data["actions"][agent]),
            pd.DataFrame(value.data["acts"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["actions"][agent])), order='F').max(axis=1), columns=value.data["actions"][agent])
        ] for key, value in self.data.items()}
        
        fig = go.Figure()
        
        if on == 'all':
            for key, color in zip(collector.keys(), COLORS, strict=False):
                for action in collector[key][0].columns:
                    fig.add_trace(go.Scatter(x=collector[key][0].index, y=collector[key][0][action], mode='lines', line={"color": color}, legendgroup=str(key)+str(action), name=str(key)+"_"+str(action)))
                    fig.add_trace(go.Scatter(x=collector[key][1].index.tolist() + collector[key][2].index[::-1].tolist(),
                                             y=collector[key][1][action].tolist() + collector[key][2][action][::-1].tolist(),
                                             fill='toself',
                                             fillcolor='rgba' + str(pc.hex_to_rgb(color) + (0.2,)),
                                             name=key,
                                             legendgroup=str(key)+str(action),
                                             showlegend=False,
                                             line=dict(color='rgba' + str(pc.hex_to_rgb(color) + (0,)))))
        else:
            for index, key in enumerate(collector.keys()):
                fig.add_trace(go.Scatter(x=collector[key][0].index, y=collector[key][0][on], mode='lines', legendgroup=key, name=key, line={"color": COLORS[index]}))
                fig.add_trace(go.Scatter(x=collector[key][1].index.tolist() + collector[key][2].index[::-1].tolist(),
                                         y=collector[key][1][on].tolist() + collector[key][2][on][::-1].tolist(),
                                         fill='toself',
                                         fillcolor='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0.2,)),
                                         name=key,
                                         legendgroup=key,
                                         showlegend=False,
                                         line=dict(color='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0,)))))

        fig.show()
    
    def make_actions_movie(self, agent: str, on: list[str]) -> None:
        # Collect data for each key in self.data
        collector = {
            key: [
                pd.DataFrame(
                    value.data["acts"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["actions"][agent])), order='F').mean(axis=1),
                    columns=value.data["actions"][agent]
                ),
                pd.DataFrame(
                    value.data["acts"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["actions"][agent])), order='F').min(axis=1),
                    columns=value.data["actions"][agent]
                ),
                pd.DataFrame(
                    value.data["acts"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["actions"][agent])), order='F').max(axis=1),
                    columns=value.data["actions"][agent]
                )
            ]
            for key, value in self.data.items()
        }

        frames = []
        for key in collector.keys():
            data = []
            for index, action in enumerate(on):
                data.append(go.Scatter(
                    x=collector[key][0].index,
                    y=collector[key][0][action],
                    mode='lines',
                    legendgroup=action,
                    name=action,
                    line={"color": COLORS[index]}
                ))
                data.append(go.Scatter(
                    x=collector[key][1].index.tolist() + collector[key][2].index[::-1].tolist(),
                    y=collector[key][1][action].tolist() + collector[key][2][action][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0.2,)),
                    name=action,
                    legendgroup=action,
                    showlegend=False,
                    line=dict(color='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0,))),
                ))
            frames.append(go.Frame(data=data,
                                   layout=go.Layout(title=
                                                    {"text": key,
                                                     "x": 0.5,
                                                     "font": {"size": 30, 'family': 'Arial'}
                                                     },
                                                    font={"size": 30, 'family': 'Arial'},
                                                    )))
                
        initial_data = []
        
        for index, action in enumerate(on):
            initial_data.append(
                go.Scatter(
                    x=collector[list(collector.keys())[0]][0].index,
                    y=collector[list(collector.keys())[0]][0][action],
                    mode='lines',
                    legendgroup=action,
                    name=action,
                    line={"color": COLORS[index]}
                )
            )
            initial_data.append(
                go.Scatter(
                    x=collector[list(collector.keys())[0]][1].index.tolist() + collector[list(collector.keys())[0]][2].index[::-1].tolist(),
                    y=collector[list(collector.keys())[0]][1][action].tolist() + collector[list(collector.keys())[0]][2][action][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0.2,)),
                    name=action,
                    legendgroup=action,
                    showlegend=False,
                    line=dict(color='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0,))),
                )
            )

        fig = go.Figure(
            data=initial_data,
            layout=go.Layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[None]
                            )
                        ]
                    )
                ]
            ),
            frames=frames
        )
        fig.update_layout(
            title_text="Iteration: 0",
            title_x=0.5,
            font={"size": 30, 'family': 'Arial'},
        )
        fig.show()
    
        return fig
    
    def make_states_movie(self, agent:str, on:list[str])->None:
        collector = {key: [
            pd.DataFrame(value.data["obs"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["states"][agent])), order='F').mean(axis=1), columns=value.data["states"][agent]),
            pd.DataFrame(value.data["obs"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["states"][agent])), order='F').min(axis=1), columns=value.data["states"][agent]),
            pd.DataFrame(value.data["obs"][agent].numpy().reshape((-1, value.data["batches_per_episode"], len(value.data["states"][agent])), order='F').max(axis=1), columns=value.data["states"][agent])
        ] for key, value in self.data.items()}
        
        frames = []
        for key in collector.keys():
            data = []   
            for index, state in enumerate(on):
                data.append(go.Scatter(
                    x=collector[key][0].index,
                    y=collector[key][0][state],
                    mode='lines',
                    legendgroup=state,
                    name=state,
                    line={"color": COLORS[index]}
                ))
                data.append(go.Scatter(
                    x=collector[key][1].index.tolist() + collector[key][2].index[::-1].tolist(),
                    y=collector[key][1][state].tolist() + collector[key][2][state][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0.2,)),
                    name=state,
                    legendgroup=state,
                    showlegend=False,
                    line=dict(color='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0,))),
                ))
            frames.append(go.Frame(data=data,
                                   layout=go.Layout(title=
                                                    {"text": key,
                                                     "x": 0.5,
                                                     "font": {"size": 30, 'family': 'Arial'}
                                                     },
                                                    font={"size": 30, 'family': 'Arial'},
                                                    )))
                
        initial_data = []
        
        for index, state in enumerate(on):
            initial_data.append(
                go.Scatter(
                    x=collector[list(collector.keys())[0]][0].index,
                    y=collector[list(collector.keys())[0]][0][state],
                    mode='lines',
                    legendgroup=state,
                    name=state,
                    line={"color": COLORS[index]}
                )
            )
            initial_data.append(
                go.Scatter(
                    x=collector[list(collector.keys())[0]][1].index.tolist() + collector[list(collector.keys())[0]][2].index[::-1].tolist(),
                    y=collector[list(collector.keys())[0]][1][state].tolist() + collector[list(collector.keys())[0]][2][state][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0.2,)),
                    name=state,
                    legendgroup=state,
                    showlegend=False,
                    line=dict(color='rgba' + str(pc.hex_to_rgb(COLORS[index]) + (0,))),
                )
            )

        fig = go.Figure(
            data=initial_data,
            layout=go.Layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[None]
                            )
                        ]
                    )
                ]
            ),
            frames=frames
        )
        fig.update_layout(
            title_text="Iteration: 0",
            title_x=0.5,
            font={"size": 30, 'family': 'Arial'},
        )
        fig.show()
        
        return fig
    

        
        
    
if __name__ == "__main__":
    cmp=CompareDataProcessor({
        "Trained":DataProcessor(f"/Users/parsaghadermarzi/Desktop/Academics/Projects/SPAM-DFBA/examples/Results/Bacillus_168_Xylan/Bacillus_168_Xylan_2500.pkl")  ,
        "DFBA":DataProcessor(f"/Users/parsaghadermarzi/Desktop/Academics/Projects/SPAM-DFBA/examples/Results/Bacillus_168_XylanـNoControl/Bacillus_168_XylanـNoControl_0.pkl")  
    })
    cmp.compare_state("Bacllus_agent1",on="Bacllus_agent1")
    # cmp.make_states_movie("Bacllus_agent1",on=['Bacllus_agent1', 'xyl__D_e', 'Xylan'])
    # cmp.make_actions_movie("Bacllus_agent1",on=['xylanase_production', 'xylosidase_production'])

    
    
    
    