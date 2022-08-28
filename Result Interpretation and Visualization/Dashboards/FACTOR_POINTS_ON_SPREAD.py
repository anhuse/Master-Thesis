# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:22:09 2022

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
import yfinance as yf
from datetime import datetime

import matplotlib.pyplot as plt
import statsmodels.api as sm

### GLoabal variables
START = '2019-01-01'
END = datetime.today()
INTERVAL = '1d'
WINDOW_SIZE = 5
QUANTILE = 0.95

#%% Function

def get_oil_and_pair(tickers: str, start: str, end: str, interval: str
                     ) -> pd.DataFrame:
    """
    

    Parameters
    ----------
    tickers : str
        string listing the tickers to investigate.
    start : str
        start og the interval.
    end : str
        end of the interval.
    interval : str
        lenght of the interval.

    Returns
    -------
    data : pandas.DataFrame
        Close and Volume data for the tickers specified (Close in pct change)

    """
    
    data = yf.download(tickers + ', BZ=F', start, end, interval, rounding=True)\
           [['Close', 'Volume']]
    data['Close'] = data['Close'].pct_change()
    data = data[1:]
    return data

#%%
pairs = ['ELK.OL WCH.DE',
         'ORK.OL BN.PA',
         'EQNR.OL LUNE.ST']


#%% APP
app = dash.Dash(__name__)


app.layout = html.Div([
    html.Div([
        ### left side
        html.H1('Pairs - Dashboard'),
        dcc.Markdown("""
                  This dashboard investigates if the development of a pair's spread
                  coincides with large relative changes in the oil price, regarding
                  percentage change in close price, std, and large traded volumnes
                  """),
        
        dcc.Dropdown(
        id = 'pairs-selector',
        options = [{
                      'label': '{} and {}'.format(i.split(' ')[0],
                                                  i.split(' ')[1]),          
                      'value': i} for i in pairs],
        value = pairs[1]),
        
        dcc.Markdown("""
                     ### Investigating the ratio: 
                     """),
        
        dcc.Graph(id='ratio-oil-graph'),
        dcc.Markdown("""
                     #### Factors to include: 
                     """),
        
        dcc.RadioItems(
            id = 'factors-to-include',
            options=[{'label': v, 'value': i} for i, v in {1: 'pct_change', \
                                                           2: 'std',
                                                           3: 'volume'}.items()],
            value = 1,
            labelStyle = {'display': 'inline-block'}
    )
        
        
        
        ])])
        
                     
@app.callback(
    Output('ratio-oil-graph', 'figure'),
    Input('pairs-selector', 'value'),
    Input('factors-to-include', 'value')
    )
def update_pairs_graph(pair):
    
    









if __name__ in '__main__':
    app.run_server()
    
    
    
    
    