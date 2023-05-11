import pandas as pd
from evoviewpandas import EvoViewPandas
import streamlit as st
import utils

# Configure page
st.set_page_config(
    page_title="EvoView",
    page_icon="🧬",
    layout="wide",
)

utils.config_plots()

# Constants
NUM_COLS = 5
USER_RATINGS_SESSION_KEY = 'user_ratings'
POPULATION_SESSION_KEY = 'population'
HOF_SESSION_KEY = 'hof'
LOGGER_SESSION_KEY = 'logger'
INTERACTION_COUNT = 'interaction_count'

POPULATION_SIZE = 50
MAX_GENERATIONS = 5
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.2
NUM_VIEWS_TO_SHOW = 10
USE_FITNESS_SHARING = True

## TODO: Create a new EvoView object when hypeparameter sliders change
# with st.sidebar:
#     st.markdown('# 🧬 Genetic Algorithm Hyperparameters')

#     POPULATION_SIZE = st.slider(label = '#### Population Size:',
#                                   min_value = 10,
#                                   max_value = 100,
#                                   step = 1,
#                                   value = 50)
    
#     MAX_GENERATIONS = st.slider(label = '#### Max. Generations:',
#                                   min_value = 1,
#                                   max_value = 25,
#                                   step = 1,
#                                   value = 5)

#     CROSSOVER_RATE = st.slider(label = '#### Crossover Rate (CXR):',
#                                   min_value = 0.0,
#                                   max_value = 0.8,
#                                   step = 0.1,
#                                   value = 0.7)
    
#     MUTATION_RATE = 1 - CROSSOVER_RATE - 0.1

#     st.markdown(f'#### Mutation Rate (MTR): {MUTATION_RATE:.1f}')
#     st.markdown(f'#### Reproduction Rate (RPX): {1 - MUTATION_RATE - CROSSOVER_RATE:.1f}')

#     st.markdown('# 🎛️ View Panel Configuration')

#     NUM_VIEWS_TO_SHOW = st.slider(label = '#### Number of Views to Show:',
#                                   min_value = 10,
#                                   max_value = min(POPULATION_SIZE, 30),
#                                   step = 1,
#                                   value = 10)
    
st.markdown('# 🧬 EvoView: Evolving Views for Data Exploration')

@st.cache_data
def get_data() -> pd.DataFrame:
    df, view_space = utils.load_nba_data()
    return df, view_space

dataframe, view_space = get_data()

evoview = EvoViewPandas(
            dataframe, 
            view_space, 
            population_size=POPULATION_SIZE,
            top_views_size=NUM_VIEWS_TO_SHOW,
            use_fitness_sharing=USE_FITNESS_SHARING,
            verbose=1)

if not LOGGER_SESSION_KEY in st.session_state:
    st.session_state[LOGGER_SESSION_KEY] = utils.create_logger()

logger = st.session_state[LOGGER_SESSION_KEY]

if not INTERACTION_COUNT in st.session_state:
    st.session_state[INTERACTION_COUNT] = 0

# Saving current hall of fame
if not HOF_SESSION_KEY in st.session_state:
    hof = evoview.hall_of_fame
    st.session_state[HOF_SESSION_KEY] = hof

hof = st.session_state[HOF_SESSION_KEY]

# Saving current population
if not POPULATION_SESSION_KEY in st.session_state:
    st.session_state[POPULATION_SESSION_KEY] = evoview.population

evoview.population = st.session_state[POPULATION_SESSION_KEY]

average_fitness, std_fitness = evoview.compute_average_std_fitness(hof)
pop_diversity = evoview.compute_population_diversity(hof)

st.markdown('#### Metrics')

col1, col2, col3, col4 = st.columns(4, gap='small')

col1.metric(
    label="Avg. Fitness (high is better)",
    value=f'{average_fitness:.3f}',
    delta=None,
    delta_color='normal',
)

col2.metric(
    label="Std. Fitness (low is better)",
    value=f'{std_fitness:.3f}',
    delta=None,
    delta_color='normal',
)

col3.metric(
    label="Avg. Diversity (high is better)",
    value=f'{pop_diversity:.1f}',
    delta=None,
    delta_color='normal',
)

with col4:
    if st.button('🧬 Evolve'):
        hof = evoview.evolve(generations=MAX_GENERATIONS)
        st.session_state[HOF_SESSION_KEY] = hof
        st.session_state[POPULATION_SESSION_KEY] = evoview.population
        st.session_state[INTERACTION_COUNT] += 1
        logger.info({ 'round' : st.session_state[INTERACTION_COUNT],  
                   'avgfitness' : average_fitness, 
                   'stdfitness' : std_fitness, 
                   'hofdiversity' : pop_diversity })

st.markdown('#### Most Interesting Views')

views = hof
print(views)

cols = list(st.columns(NUM_COLS, gap='medium'))

c = 0
for i, v in enumerate(views):
    if i % NUM_COLS == 0:
        c = 0

    with cols[c]:
        df_binned_views = evoview.get_binned_normalized_views(v)
        fig = utils.plot_binned_view_grouped(df_binned_views, 'Bin', 'Normalized')
        st.pyplot(fig)
        subcol1, subcol2 = st.columns(2, gap='large')
        with subcol1:
            if st.button('🖒 Like', key=f'like_{i}'):
                if not USER_RATINGS_SESSION_KEY in st.session_state:
                    st.session_state[USER_RATINGS_SESSION_KEY] = dict()
                
                st.session_state[USER_RATINGS_SESSION_KEY][str(v)] = 1.5

        with subcol2:
            if st.button('🖓 Dislike', key=f'dislike_{i}'):
                if not USER_RATINGS_SESSION_KEY in st.session_state:
                    st.session_state[USER_RATINGS_SESSION_KEY] = dict()

                st.session_state[USER_RATINGS_SESSION_KEY][str(v)] = 0.2

    c += 1

if USER_RATINGS_SESSION_KEY in st.session_state:
    evoview.get_user_feedback(st.session_state[USER_RATINGS_SESSION_KEY])
