import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import time
from evoview import ViewSpace

def load_nba_data():
    sys.path.append('..')
    dataset_path = os.path.join("data/raw","Seasons_Stats.csv")

    df = pd.read_csv(dataset_path, skipinitialspace=True, index_col=0)
    df_source = df[df['Year'] == 2015]
    
    cols = df_source.columns
    df_source[cols] = df_source[cols].apply(pd.to_numeric, errors='ignore', axis=1)

    df_source = df_source.replace(["\n","NaN"], np.nan)
    df_source = df_source.replace(np.nan, 0)
    df_source = df_source.applymap(lambda s:s.lower() if type(s) == str else s)

    view_space= ViewSpace(agg_functions=['sum', 'count', 'mean'], 
                          measure_attributes=['AST', '2P', '2PA', '3P', '3PA', '3PAr','FG', 'FGA','FT', 'FTA', 'FTr' ,'PTS', 'TRB', 'STL', 'BLK', 'DRB', 'ORB',],
                    dimension_attributes=['Age','G', 'GS', 'MP'])


    return df_source, view_space

def config_plots():
    # Set global matplotlib parameters
    mpl.rcParams['font.size'] = 14

    # Remove plot edges
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.grid.axis'] = 'y'
    mpl.rcParams['grid.alpha'] = 0.3

    plt.style.use('ggplot')

def plot_binned_view_grouped(df,x_axis,y_axis):
    COLORS = ['#1b9e77', '#7570b3']
    fig = plt.figure()
    ax_bvg = sns.barplot(x=x_axis, y=y_axis, data=df, hue='View_Type', palette=COLORS, width=0.8)
    a_attr = df['Dimension'].unique()[0]
    m_attr = df['Measure'].unique()[0]
    aggr = df['Function'].unique()[0]
    ax_bvg.set_xlabel(a_attr)
    ax_bvg.set_ylabel(f'Prob. distr. of {aggr}({m_attr})')
    plt.legend(frameon=False, framealpha=0.5)
    return fig


def create_logger():
    filename = f'EvoView_{time.strftime("%Y%m%d-%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=filename),
        ]
    )

    return logging.getLogger("logger")