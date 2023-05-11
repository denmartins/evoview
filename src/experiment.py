from evoviewpandas import EvoViewPandas
from utils import load_nba_data

def simulate_user_feedback(best_views):
    user_ratings = dict()
    for c, v in enumerate(best_views):
        if c < 3:
            user_ratings[str(v)] = 2
        else:
            user_ratings[str(v)] = 0.1
    return user_ratings

def main():
    df, vs = load_nba_data()

    for seed in [1235, 42, 16]:
        evolution = EvoViewPandas(df, vs, 
                                population_size=100,
                                top_views_size=10, 
                                use_fitness_sharing=True,
                                verbose=0, 
                                random_seed=seed)
        
        for i in range(10):
            hof = evolution.evolve(generations=20)
            ## Uncomment the two lines below to incorporate (simulated) user feedback
            user_ratings = simulate_user_feedback(hof)
            evolution.get_user_feedback(user_ratings)
            f,s = evolution.compute_average_std_fitness(hof, raw_fitness=1)
            results = [i, f, s, evolution.compute_population_diversity(hof)]
            
            result = ';'.join(map(str, results))
            print(result)


if __name__ == '__main__':
    main()