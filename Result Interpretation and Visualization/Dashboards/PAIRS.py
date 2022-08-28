# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 11:04:31 2022

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
from datetime import datetime

import yfinance as yf

import matplotlib.pyplot as plt

#%% predefined

pairs = ['ELK.OL WCH.DE',
         'ORK.OL BN.PA',
         'EQNR.OL LUNE.ST']

def pair_df(pair: str, start: str, end: datetime, rebased = False) -> pd.DataFrame:
    """

    Parameters
    ----------
    pair : str
        A string composed of two tickers, forming the pair
    start : str
        Start of time series
    end : datetime object
        End of time series
    rebased : Bool
        If or if not the columns are to be rebased to 100

    Returns
    -------
    df : pd.DataFrame
        DataFrame with two columns, namely the close prices of the securities

    """
    df = yf.download(pair, start, end)['Close']
    if rebased:    
        df = df / df.iloc[0] + 100
    
    return df


def z_score(series):
    """
    Calculates z-scores for a given timeseries
    
    Parameters
    -----------
    series : Pandas timeseries
        time series
        
    Returns
    -----------
    Pandas timeseries
        z-scores of series
    """
    return (series - series.mean()) / np.std(series)

#%% curreny IPO
fig_curr = go.Figure()
fig_curr.add_trace(go.Indicator(
        mode = 'number',
        title = {'text': 'currency'},
        value = 9.657))
fig_curr.update_layout(paper_bgcolor = "lightgreen",
                       height=100,
                       width=100)


#%% Building dashboard
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
    html.Div([
        ### left side
        html.H1('Pairs - Dashboard'),
        dcc.Markdown("""### Description of dashboard and functionality: \n
                         Section1:               \n
                         Section2:               \n
                        
                     """
                          )], style = {'width': '48%',
                                       'display': 'inline-block'}),
    
        ### right side
        html.Div([
            dcc.Graph(id = 'currency-IPO',
                      figure = fig_curr),
            

        ], style = {'width': '48%', 'float': 'right', 'display': 'inline-block'})
        
        ]),
    
    html.H3('Choose pair: '),
    dcc.Dropdown(
        id = 'pairs-selector',
        options = [{
                      'label': '{} and {}'.format(i.split(' ')[0],
                                                  i.split(' ')[1]),          
                      'value': i} for i in pairs],
        value = pairs[1]
        
        ),
    dcc.Graph(id='pair-graph'),
    
    html.H3('Choose a bandwidth for the below graph: '), 

    dcc.Input(
        id = 'band-width',
        type = 'number',
        placeholder = 'Band width: ',
        value=2
        ),
    
    dcc.Graph(id='pair-z-score')
    
    ])
                 

@app.callback(
    Output('pair-graph', 'figure'),
    Input('pairs-selector', 'value')
    )
def update_pairs_graph(pair):
    
    df = pair_df(pair, '2020-01-01', datetime.today(), rebased=True)
    p1 = df.columns[0]
    p2 = df.columns[1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df.index,
                             y = df[p1],
                             mode = 'lines',
                             name = p1
                            ))
    fig.add_trace(go.Scatter(x = df.index,
                             y = df[p2],
                             mode = 'lines',
                             name = p2
                            ))
    fig.update_layout(title = 'Comparison of pair-securities',
                      legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1)
                      )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    return fig
    
    
@app.callback(
    Output('pair-z-score', 'figure'),
    Input('pairs-selector', 'value'),
    Input('band-width', 'value')
    )
def update_z_graph(pair, value):
    
    df = pair_df(pair, '2020-01-01', datetime.today(), rebased=False)
    p1 = df.columns[0]
    p2 = df.columns[1]
    
    ratio = df[p1] / df[p2]
    ratio_z = z_score(ratio)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df.index,
                             y = ratio_z,
                             mode = 'lines',
                             name ='z-score'
                            ))
    ### Horisontal lines
    fig.add_hline(y=value, line_color='red', line_dash='dash',
                  annotation_text = 'upper border')
    fig.add_hline(y=3, line_color='black', opacity=0.8,
                  annotation_text = 'outside region')
    fig.add_hline(y=0, line_color='black', opacity=0.8)
    fig.add_hline(y=-value, line_color='green', line_dash='dash',
                  annotation_text = 'lower border',
                  annotation_position = 'bottom right')
    fig.add_hline(y=-3, line_color='black', opacity=0.8,
                  annotation_text = 'outside region',
                  annotation_position = 'bottom right')
    
    ### Filling
    fig.add_hrect(y0=3, y1=4, fillcolor='red', opacity=0.2)
    fig.add_hrect(y0=-3, y1=-4, fillcolor='red', opacity=0.2)
    
    ### Layout
    fig.update_layout(title = 'z-score with boundaries',
                      showlegend=True)
    
    fig.update_layout(
        title = 'z-score with boundaries',
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
))
    
    ### Get pairs
    
    
    return fig


    
if __name__ in '__main__':
    app.run_server()