# ### Creating Views
# 
# - The **Target View** vTi represents a view with the same set of triple (a, m, f) applied to a subset of the data DQ that is produced by a given user query Q.
# 
#     SELECT MP, SUM(3PAr) 
#     FROM players
#     WHERE team=GSW AND year=2015 
#     GROUP BY MP
# 
#     * (a, m, f)
# 
#         * a = MP
#         * m = 3PAr
#         * f = sum
#         * team=GSW
# 
# 
# - The **Reference View** vRi visualizes the results of grouping the data in the whole database D by a, and then aggregating the values in m with f
# 
#     SELECT MP, SUM (3PAr) 
#     FROM players
#     WHERE year=2015 
#     GROUP BY MP
# 
#     * (a, m, f)
#         * a = MP
#         * m = 3PAr
#         * f = sum
#         
#         
# - The **Binning View** is for binning the two views to show some very interesting observation
# 
#     SELECT A, F(M) FROM DB WHERE T
#     GROUP BY A
#     NUMBER OF BINS b
# 

import numpy as np
import pandas as pd

def create_aggr_view(dataframe, dimension, measure, function):
    return dataframe.groupby(dimension).agg(**{function:pd.NamedAgg(measure, function)}).reset_index()
    
def create_target_view(dataframe, dimension, measure, function, subset_attr, subset_value):
    df_target_view = dataframe[dataframe[subset_attr] == subset_value]
    return create_aggr_view(df_target_view, dimension, measure, function)

def create_reference_view(dataframe, dimension, measure, function):
    return create_aggr_view(dataframe, dimension, measure, function)

def create_binned_view(dataframe, df_column, cut_bins, cut_labels):
    dataframe['Bin'] = pd.cut(dataframe[df_column], bins=cut_bins, labels=cut_labels)
    df_binned = create_aggr_view(dataframe, 'Bin', df_column, 'sum')
    return df_binned


def normalize_values(df_column):
    return df_column/(df_column.sum())