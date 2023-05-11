from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class View:
    dimension_attribute:str
    measure_attribute:str
    aggretation_function:str
    
class ViewSpace:
    def __init__(self, dimension_attributes:List[str], measure_attributes:List[str], agg_functions:List[str]) -> None:
        self.dimension_attributes = dimension_attributes 
        self.measure_attributes = measure_attributes 
        self.agg_functions = agg_functions
        self.views = self.construct_all_views()

    def construct_all_views(self):
        """Constructs a set of all candidate view combinations without materializing views.

        Returns:
            Set: Set of all candidate views.
        """
        self.views = set([(a,m,f) 
                          for a in self.dimension_attributes 
                          for m in self.measure_attributes
                          for f in self.agg_functions])

        return self.views
    
  
class EvoView(ABC):
    def __init__(self, view_space:ViewSpace) -> None:
       self.view_space = view_space
    
    @abstractmethod
    def setup_evolution(self):
        """Configures evolutionary process and operators.
        """
        raise NotImplementedError()
     
    @abstractmethod
    def get_user_feedback(self, view_rating_dict):
        """Incorporates subjective evaluation via user rating.

        Args:
            view_rating_dict (dictionary): dictionary in the form of {view (str) : user_rating (float)}
        """
        raise NotImplementedError()

    @abstractmethod
    def evolve(self, generations:int):
        """Runs a genetic algorithm for evolving views over a certain number of generations. 


        Args:
            generations (int): number of generations for evolving views
        """
        raise NotImplementedError()
        


    
