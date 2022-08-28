# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:06:49 2022

@author: Anders Huse
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.pyplot as plt



#%% Performing OPTICS clustering

### Generate sample data

np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + 0.8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + 0.1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + 0.3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)

X = np.vstack((C1, C2, C3, C4, C5, C6))


#%% Building dashboard
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('OPTICS demo, changing hyperparameter values'),
    dcc.Markdown("""### Description of the differemt OPTICS-parameters: \n
                  min_samples: The number of samples in a neighborhood for
                  a point to be considered as a core point.    \n 
                  xi: Determines the minimum steepness on the reachability plot
                  that constitutes a cluster boundary.     \n 
                  min_cluster_size: Minimum number of samples in an 
                  OPTICS cluster    \n """
                      )
             ,
             
    dcc.Graph(id='optics-graph'),

    dcc.Markdown('min_samples:'),
    dcc.Slider(
        id = 'min-sample-slider',
        min = 20,
        max = 80,
        value = 50,
        marks = {str(i): str(i) for i in np.arange(20, 80, 5)},
        step = None
        ),            
                 
    dcc.Markdown('xi:'),
    dcc.Slider(
        id = 'xi-slider',
        min = 0.001,
        max = 0.05,
        value = 0.05,
#        marks = {str(i): str(i) for i in np.arange(0.001, 0.055, 0.005)},
        marks = {i: '{:.3f}'.format(i) for i in np.arange(0.001, 0.055, 0.005)},
        step = None
        ),

    dcc.Markdown('min_cluster_size:'),
    dcc.Slider(
        id = 'min-cluster-size-slider',
        min = 0.01,
        max = 0.09,
        value = 0.05,
#        marks = {str(i): str(i) for i in np.arange(0.01, 0.1, 0.01)},
        marks = {i: '{:.3f}'.format(i) for i in np.arange(0.01, 0.1, 0.01)},
        step = None
        ),
        
    dcc.Markdown("""Type of distance calculation"""),
    dcc.RadioItems(
        id = 'distance-method',
        options=[{'label': v, 'value': i} for i, v in {1: 'manhattan', \
                                                       2: 'eucledian'}.items()],
        value = 1,
        labelStyle = {'display': 'inline-block'}
    )
    
        ])
    

@app.callback(
    Output('optics-graph', 'figure'),
    Input('xi-slider', 'value'),
    Input('min-cluster-size-slider', 'value'),
    Input('min-sample-slider', 'value'),
    Input('distance-method', 'value')
             )

def update_graph(x_i, min_cluster, min_sample, p):
    
    ### re-making/ fitting
    clust = OPTICS(min_samples=min_sample, xi=x_i,\
                   min_cluster_size=min_cluster, p=p)
    clust.fit(X)
    
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    
    fig = go.Figure()
    
    for klass in range(0,5):
        
        Xk = X[clust.labels_ == klass]
        
        fig.add_trace(go.Scatter(x=Xk[:, 0],
                                 y=Xk[:, 1],
                                 mode = 'markers',
                                 opacity=1,
                                 name='Cluster ' + str(klass),
                                 marker = dict(colorscale='Viridis')
                                 
                                 ))
    fig.add_trace(go.Scatter(x = X[clust.labels_ == -1, 0],
                             y = X[clust.labels_ == -1, 1],
                             mode = 'markers',
                             name = 'Non-cluster points',
                             opacity = 0.3,
                             
                             ))
    fig.update_layout(title = 'Automatic Clustering OPTICS',
                      yaxis_zeroline=False,
                      xaxis_zeroline=False)
    
    return fig
    
    
if __name__ == '__main__':
    app.run_server()
        
        
        