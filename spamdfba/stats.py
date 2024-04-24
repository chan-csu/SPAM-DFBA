"""This module is designed for statistical analysis of the simulation results."""
import numpy as np 
import pandas as pd
import pickle
import plotly.express as px
import scipy.stats as stats

class StatResult:
    """
    This class is designed for presenting the results of statistical analyses offered by this, stats, module.
    Two vectors of observations are required to be compared. They can vary in size but it would be strange if this 
    is the case with SPAM-DFBA simulations.

    Args:
        vec1 (np.ndarray): First vector of observations.
        vec2 (np.ndarray): Second vector of observations.
        names (list[str]): Names of the two vectors in the same order as they are passed as vec1 and vec2.
    
    """
    def __init__(self, vec1: np.ndarray,vec2:np.ndarray, names: list[str]):
        self.vec1 = vec1
        self.vec2 = vec2
        self.names = names
        self.mean1 = np.mean(vec1)
        self.mean2 = np.mean(vec2)
        self.std1 = np.std(vec1)
        self.std2 = np.std(vec2)
        
    
    def box_plot(self, plot:bool=True)->None:
        """
        A simple box plot visualization of the two vectors of observations.
        """
        df=pd.DataFrame({
            self.names[0]:self.vec1,
            self.names[1]:self.vec2,
        })
        df=pd.melt(df,value_vars=['obs1','obs2'])
        fig=px.box(df,x='variable',y='value',color='variable')
        if plot:
            fig.show()
        return fig
    
    def anova(self):
        """
        Performs one-way ANOVA test on the two vectors of observations.
        """
        return stats.f_oneway(self.vec1,self.vec2)
    
    def __str__(self):
        ares=self.anova()
        return f"""
    
    mean1:{self.mean1}
    mean2:{self.mean2}
    std1:{self.std1}
    std2:{self.std2}
    -------------------
    ANOVA: statistic={ares.statistic}, pvalue={ares.pvalue}
    """
        
    


def compare_observations(
                        obs1: np.ndarray,
                        obs2: np.ndarray,
                        compounds: list[int],
                        on_index: int,
                        agent:str='agent1',
                        )-> list[StatResult]:
    """Performs statistical analysis of two batches of observations.
    Args:
        obs1: address of the first batch of observations save as .pkl file.
        obs2: address of the second batch of observations save as .pkl file.
        compounds: List of compounds to compare.
        on_index: Index of the compound to compare.
    Returns:
        (StatResult): Statistical analysis results represented as a StatResult object.
    """

    with open(obs1, "rb") as f:
        obs1 = pickle.load(f)

    with open(obs2, "rb") as f:
        obs2 = pickle.load(f)
    
        
    results=[]
    obs1=obs1[agent]
    obs2=obs2[agent]
    obs1=obs1[:,compounds]
    obs2=obs2[:,compounds]
    
    for i in range(len(compounds)):
        temp1=obs1[:,i].reshape(-1,4).numpy()[on_index,:]
        temp2=obs2[:,i].reshape(-1,4).numpy()[on_index,:]
        results.append(StatResult(temp1,temp2,["obs1","obs2"]))
    
    return results
    
    
    
    
    
    
    
if __name__ == "__main__":
    pass
    
            