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

import statsmodels.api as sm
import matplotlib.pyplot as plt

from PAIRS import z_score

### GLoabal variables
START = '2019-01-01'
END = datetime.today()
INTERVAL = '1d'
WINDOW_SIZE = 5
QUANTILE = 0.95
PAIRS = ['ELK.OL WCH.DE',
         'ORK.OL BN.PA',
         'EQNR.OL LUNE.ST']

#%% Functions
### Some of the below functions are from the "framework" file, but are modified
### to accomodate the specific purpose of this dashboard

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


def spread(S1: pd.Series, S1_name: str, S2: pd.Series) -> pd.Series:
    """
    Calculates the spread of the two series' return Close-price Series

    Parameters
    ----------
    S1 : pandas.Series
        Close prices for the first company in the pair.
    S2 : pandas.Series
        Close prices for the second company in the pair.
    S1_name : str
        Name of the ticker of the first series

    Returns
    -------
    spread : pandas.Series
        The calculated spread.

    """

    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    S1 = S1[S1_name]
    b = results.params[1]
    spread = S2 - b * S1
        
    return spread
 

def get_oil_data(ticker: str, start: str, end: str, interval: str, window: int,
                 QUANTILE: float) -> pd.DataFrame:
    """
    
    
    Parameters
    ----------
    ticker : str
        string identifying what oil-price to investigate.
    start : str
        start og the interval.
    end : str
        end of the interval.
    interval : str
        lenght of the interval.
        
    window : int
        Window size of rolling stats
        
    QUANTILE : float
        quantile to define cut-off for extreme points

    Returns
    -------
    df : pandas.DataFrame
        Close and Volume df for the tickers specified (Close in pct change)

    """
    
    data = yf.download(ticker, start, end, interval,rounding = True)\
         [['Close', 'Volume']]


    data['Close'] = data['Close'].pct_change()
    data = data.fillna(method='ffill')
    data = data[1:]
    
    ### Assigning rolling columns
    data = data.assign(rolling_pct = data['Close'].rolling(window).mean(),
                       rolling_vol = data['Volume'].rolling(window).mean(),
                       rolling_std = data['Close'].rolling(window).std()
                      )
    ### Setting cut-off points for "extreme points"
    
    pct_bound = data['rolling_pct'][window:].quantile(QUANTILE)
    std_bound = data['rolling_std'][window:].quantile(QUANTILE)
    vol_bound = data['rolling_vol'][window:].quantile(QUANTILE)

    ### Assigning new "extreme" boolean points
    data = data.assign(pct_level = np.where(abs(data['rolling_pct']) \
                                                > pct_bound, 1, 0),
                       vol_level = np.where(data['rolling_vol'] > vol_bound, 1, 0),
                       std_level = np.where(data['rolling_std'] > std_bound, 1, 0)
                      )
    data.head()

    
    return data

def close_prices(tickers: str, start: str, end: str,
                 interval: str, df_main: pd.DataFrame) -> pd.DataFrame:
    """
    Attains the close prices of the ticker in the pair

    Parameters
    ----------
    tickers : str
        str of tickers to which data is downloaded from.
    start : str
        start of interval to investigate.
    end : str
        end of interval to investigate.
    interval : str
        interval granularity.
    df_main : pd.DataFrame
        pandas DataFrame whos index should be set as standard, for easy comparison.

    Returns
    -------
    data_close : pandas.DataFrame
        Close prices for the specified tickers

    """
    
    
    data_close = yf.download(tickers, start, end, interval, rounding = True)\
         [['Close', 'Volume']]

    data_close = data_close.reindex(df_main.index, axis=0).fillna(method='ffill')
    
    return data_close

#%% Database




#%% APP
app = dash.Dash(__name__)


app.layout = html.Div([
    html.Div([
        ### left side
        html.H1('Factors extreme points - Coherende'),
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
                      'value': i} for i in PAIRS],
        value = PAIRS[1]),
        dcc.Markdown("""
             #### Type of underlying timeseries 
             """),
        dcc.RadioItems(
            id = 'background-data',
            options = [
                {'label': 'z-score (of the ratio)', 'value': 'z-score of ratio'},
                {'label': 'spread', 'value': 'spread'}
                ],
            value='s'
            ),
        dcc.Markdown("""
                     #### Set the quantile for defining the "extreme points"
                     """),
        dcc.Input(
        id = 'Quantile',
        type = 'number',
        placeholder = 'Quantile for "Extreme points": ',
        value=0.9
        ),
        
        dcc.Graph(id='ratio-oil-graph'),
        dcc.Markdown("""
                     #### Statistics / elements of the oil price to include: 
                     """),
        
        dcc.Checklist(
            id = 'factors-to-include',
            options=[
                {'label': 'PCT change', 'value': 'pct'},
                {'label': 'STD change', 'value': 'std'},
                {'label': 'Volume', 'value': 'vol'}
                ],
            value = ['pct'],
            labelStyle = {'display': 'inline-block'}
    )
        
        
        
        ])])
        
                     
@app.callback(
    Output('ratio-oil-graph', 'figure'),
    Input('pairs-selector', 'value'),
    Input('factors-to-include', 'value'),
    Input('background-data', 'value'),
    Input('Quantile', 'value')
    )
def update_pairs_graph(pair, factors, line, q):
    
    oil_data = get_oil_data('BZ=F', START, END, INTERVAL, WINDOW_SIZE, q)
    
    p1 = pair.split(' ')[0]
    p2 = pair.split(' ')[1]
    
    close_data = close_prices(pair, START, END, INTERVAL, oil_data)
    close_data = close_data.reindex(oil_data.index, axis=0).fillna(method='ffill')
    close_data = close_data.Close
    
    ### z_score
    ratio = close_data[p1] / close_data[p2]
    z = z_score(ratio)
    z = pd.DataFrame(z, columns=['z_score']).fillna(method='ffill')
    
    ### spread
    S1 = close_data[p1]
    S2 = close_data[p2]
    s = spread(S1, p1, S2)
    s = pd.DataFrame(s, columns=['spread']).fillna(method='ffill')
    
    ### index of extreme points
    oil_pct_index = oil_data[oil_data['pct_level'] == 1].index
    oil_std_index = oil_data[oil_data['std_level'] == 1].index
    oil_vol_index = oil_data[oil_data['vol_level'] == 1].index
    
    ### z / s
    if line == 'spread':
        x, y = s.index, s['spread']
        
        ### Points --> s
        ex_points_pct = s.loc[oil_pct_index, :]
        ex_points_std = s.loc[oil_std_index, :]
        ex_points_vol = s.loc[oil_vol_index, :]
    else:
        x, y = z.index, z['z_score']
        
        ### points --> z
        ex_points_pct = z.loc[oil_pct_index, :]
        ex_points_std = z.loc[oil_std_index, :]
        ex_points_vol = z.loc[oil_vol_index, :]
    
    

    points = pd.concat([ex_points_pct, ex_points_std, ex_points_vol], axis=1)
    points.columns = ['pct', 'std', 'vol']
    
    ### Plotting    
    fig = px.line(x=x, y=y, labels={'y': f'{line}',
                                    'x': ''})
    
    ### Scatter traces
    colors = ['red', 'green', 'blue']
    for i, factor in enumerate(factors):
        fig.add_trace(go.Scatter(x=points.index,
                                 y=points[factor],
                                 marker_color = colors[i],
                                 mode='markers',
                                 name = factor))
        
    ### Updating layout
    fig.update_traces(line=dict(color='blue', width=0.9))
    # fig.update_layout(title = 'z-score with boundaries',
    #                   showlegend=True)

    return fig
    


if __name__ in '__main__':
    app.run_server()
    
    
    
    
    
    