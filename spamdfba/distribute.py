from toolkit import Simulation
import ray
from typing import Iterable

class RayTaskServer:
    def __init__(self,
                 tasks:Iterable[Simulation],
                 num_workers:int,
                 job_info:dict,
                 ):
        pass

    def validate(self):
        pass

class SlurmTaskServer:
    def __init__(self,
                 tasks:Iterable[Simulation],
                 num_workers:int,
                 job_info:dict,
                 ):
        pass

    def validate(self):
        pass

class EnvGenerator: 
    """This class is designed for building large amounts of environments for large-scale experiments.
    The base environment object should be provided to the constructor."""
    pass






   
    