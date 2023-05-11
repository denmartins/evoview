import random
import numpy as np
import pandas as pd
import viewgenerator
from evoview import EvoView, ViewSpace, View
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scipy.stats import entropy
from scipy.spatial.distance import hamming

class EvoViewPandas(EvoView):
    def __init__(self, dataframe:pd.DataFrame, view_space:ViewSpace, 
            population_size=50, 
            crossover_rate=0.7,
            mutation_rate=0.2, 
            tournament_size=5,
            top_views_size=10,
            use_fitness_sharing=True,
            verbose=0,
            random_seed=42) -> None:
        super().__init__(view_space)
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.population_size = population_size
        self.crossover_rate = crossover_rate 
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.verbose = verbose
        self.dataframe = dataframe
        self.top_views_size = top_views_size
        self.use_fitness_sharing = use_fitness_sharing
        self.logbook = None
        self.toolbox = None
        self.view_rating_dict = dict()
        self.setup_evolution()
        self.initialize_population_and_hof()

    def initialize_population_and_hof(self):
        self.population = self.toolbox.population(n=self.population_size)
        self.hall_of_fame = tools.HallOfFame(maxsize=self.top_views_size)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        self.hall_of_fame.update(self.population)

    def setup_evolution(self):
        self.toolbox = base.Toolbox()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        def _create_individuals(creator, n):
            individuals = []
            for i in range(n):
                dim = random.choice(self.view_space.dimension_attributes)
                mes = random.choice(self.view_space.measure_attributes)
                agg = random.choice(self.view_space.agg_functions)

                ind = [dim, mes, agg]
            
                individual = creator(ind)
                individuals.append(individual)
            
            return individuals

        self.toolbox.register("population", _create_individuals, creator.Individual)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxOnePoint)
        
        def _mutate_individual(individual):
            pos = random.randint(0, 2)
            space = [self.view_space.dimension_attributes, 
                     self.view_space.measure_attributes, 
                     self.view_space.agg_functions]
            
            individual[pos] = random.choice(space[pos]) 
            return individual,

        self.toolbox.register('mutate', _mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=5)

    def get_binned_normalized_views(self, individual):
        view = View(*individual)
        # constants
        SUBSET_ATTR = 'Tm'
        SUBSET_VALUE = 'gsw'
        N_CUT = 3

        # target view
        df_target_view = viewgenerator.create_target_view(self.dataframe, 
                                                        view.dimension_attribute, 
                                                        view.measure_attribute, 
                                                        view.aggretation_function, 
                                                        SUBSET_ATTR, 
                                                        SUBSET_VALUE)
        
        df_target_view['Normalized'] = viewgenerator.normalize_values(
            df_target_view[view.aggretation_function])
        df_target_view['Team'] = 'GSW'
        df_target_view['View_Type'] = 'Target'
        
        # reference view
        df_reference_view = viewgenerator.create_reference_view(self.dataframe, 
                                                                view.dimension_attribute, view.measure_attribute, 
                                                                view.aggretation_function)
        
        df_reference_view['Normalized'] = viewgenerator.normalize_values(
            df_reference_view[view.aggretation_function])
        
        df_reference_view['View_Type'] = 'Reference'

        def get_bins_and_labels(df_reference_view, df_column,n_bins):
            min_val = 1
            max_val = df_reference_view[df_column].max()
            if max_val == 1:
                min_val = 0
            interval_bins = pd.interval_range(start=min_val,end=max_val,periods=n_bins)
            cut_labels = []
            cut_bins = []
            for k in range(len(interval_bins)):
                lower_val, upper_val = interval_bins[k].left, interval_bins[k].right
                lower_val_formatted = round(lower_val,2)
                upper_val_formatted = round(upper_val,2)
                cut_bins.append(lower_val_formatted)
                cut_bins.append(upper_val_formatted)
                
                bin_formated = ''.join([str(lower_val_formatted), '-',str(upper_val_formatted)])
                cut_labels.append(bin_formated)

            
            cut_bins = list(set(cut_bins))    
            cut_bins.sort()
            
            return cut_bins, cut_labels

        CUT_BINS,CUT_LABELS = get_bins_and_labels(df_reference_view,
                                                view.dimension_attribute,
                                                N_CUT)

        df_target_binned = viewgenerator.create_binned_view(df_target_view,
                                                            view.dimension_attribute, 
                                                            CUT_BINS, 
                                                            CUT_LABELS)

        df_target_binned['Normalized'] = viewgenerator.normalize_values(
            df_target_binned['sum'])
        
        df_target_binned['View_Type'] = 'Target'
        
        df_target_binned['Dimension'] = view.dimension_attribute
        df_target_binned['Measure']   = view.measure_attribute
        df_target_binned['Function']  = view.aggretation_function
        df_target_binned['Normalized'].fillna(0, inplace=True)    

        df_reference_binned = viewgenerator.create_binned_view(df_reference_view,
                                                            view.dimension_attribute,
                                                            CUT_BINS, 
                                                            CUT_LABELS)
        
        df_reference_binned['Normalized'] = viewgenerator.normalize_values(
            df_reference_binned['sum'])

        df_reference_binned['View_Type'] = 'Reference'
        df_reference_binned['Dimension'] = view.dimension_attribute
        df_reference_binned['Measure']   = view.measure_attribute
        df_reference_binned['Function']  = view.aggretation_function
        df_reference_binned['Normalized'].fillna(0, inplace=True)

        df_binned_views = pd.concat([df_target_binned, df_reference_binned])
        return df_binned_views

    def _fitness_sharing(self, individual, other):
        distance = hamming(individual, other)

        ALPHA = 1
        SIGMA = 0.6
        
        sh = max(0, 1 - (distance/SIGMA)**ALPHA)
        return sh

    def _evaluate(self, individual):
        df_binned_views = self.get_binned_normalized_views(individual)

        def _compute_divergence(df):
            df_target = df[df['View_Type'] == 'Target']
            df_reference = df[df['View_Type'] == 'Reference']
            x = df_target['Normalized']
            y = df_reference['Normalized'] 

            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            divergence = entropy(x, y)
            return divergence

        divergence = _compute_divergence(df_binned_views)

        raw_fitness = divergence
        
        fitness = raw_fitness

        # Apply fitness sharing
        if self.use_fitness_sharing:
            fitness = fitness / (1 + sum([self._fitness_sharing(individual, p) for p in self.population]))
        
        if str(individual) in self.view_rating_dict:
            fitness = fitness * self.view_rating_dict[str(individual)]

        return fitness, raw_fitness,
    
    def get_user_feedback(self, view_rating_dict:dict):
        self.view_rating_dict = view_rating_dict

    def compute_population_diversity(self, individuals=None):
        if individuals is None: 
            individuals = self.population
        scores = [hamming(i, j) for i in individuals for j in individuals if i != j]    
        return np.mean(scores)

    def compute_average_std_fitness(self, individuals=None, raw_fitness=0):
        if individuals is None: 
            individuals = self.population
        fitnesses = [ind.fitness.values[raw_fitness] for ind in individuals]
        return np.mean(fitnesses), np.std(fitnesses)

    def evolve(self, generations=1):
        if self.population is None:
            self.population = self.toolbox.population(n=self.population_size)
        
        hof = tools.HallOfFame(self.top_views_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        new_pop, logbook = algorithms.eaSimple(
            self.population, 
            self.toolbox, 
            cxpb=self.crossover_rate, 
            mutpb=self.mutation_rate, 
            ngen=generations, 
            stats=stats, 
            halloffame=hof,
            verbose=self.verbose)

        self.population = new_pop
        self.logbook = logbook
        self.hall_of_fame = hof

        return hof