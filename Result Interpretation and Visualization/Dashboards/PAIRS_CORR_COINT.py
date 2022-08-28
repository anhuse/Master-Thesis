# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:31:44 2022

@author: Anders Huse
"""


import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import pandas as pd
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px

import sys


from Pairs_class import Pairs

#%%
def get_factors_or_indices(filename,
                sheet_name,
                skiprows: int = 3,
                start:    str ='2013-01-01',
                end:      str ='2022-01-01') -> pd.DataFrame:
    """
    Reads and cleans data for the "factors".
    Returns a dataframe with each factors as a sepreate column
    """
    
    df = pd.read_excel(filename + '.xlsx',
                         sheet_name=sheet_name,
                         skiprows=skiprows,
                         index_col='Name')
    
    df = df.drop('CURRENCY')
    
    df.index = df.index.astype(str).str[:-9]
    assert type(df.index[0]) == str, "Wrong index type"
    
    df = df[start:end]
#     df.dropna(axis=1, inplace=True)
    
    return df

#%%

COUNTRIES = ['France', 'USA', 'UK', 'Germany',  'Spain', 'Italy',
                       'Canada', 'Australia', 'Hong Kong', 'Japan', 'Swiss',
                       'Russia', 'Sweden', 'Denmark', 'Finland']

### Pairs from the different foreign exchanges:
pairs = {
    
    'Spain': {'Ebro foods - Orkla': 'ebro.mc ork.ol',
              'Acerinox, S.A - Storebrand': 'acx.mc stb.ol'
              },
    
    'Germany': {'Brenntag - Yara': 'BNR.DE yar.OL',
              'Brenntag - Gjensidige': 'BNR.DE gjf.OL'
              },
    
    'Russia': {'Gazprom - Yara': 'gazp.me yar.ol'
               }
    
    }

START = '2013-01-01'
END = '2022-01-01'

### Factors
factor_frame = get_factors_or_indices('data', 'Factors', start=START, end=END)
factor_frame.dropna(axis='columns', inplace=True)
factor_frame = factor_frame.astype(float)

assert factor_frame.isnull().sum().sum() == 0, "null-values present"

FACTORS = factor_frame.columns




#%% Styles defined

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",               # top - left
    "background-color": "#f3f3fe",
    'width': '20%',
    'display': 'inline-block',
    'fontSize': 16
    }



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


# content = html.Div(id="page-content", style=CONTENT_STYLE)


app.layout = html.Div([
        
    html.Div(
    [
    
        html.H2("Pairs Correlation and cointegration"),
        html.Hr(),
        html.P(
            "Dashboard for further exploration of pairs detected by the clustering\
            algorihtms. Cointegration within the pair and correlation with the factors\
            are in focues"
            # className='lead'
            ),
        dcc.Markdown("""
                     **Choose country**
                     """),
        dcc.Dropdown(
            id = 'country-selector',
            options = [{
                          'label': i,          
                          'value': i} for i in COUNTRIES],
            value = 'Spain'
        ),
        html.Hr(),
        dcc.Markdown("""
                     **Choose pair**
                     """
            ),
        dcc.Dropdown(
            id = 'pairs-selector',
            options = [{
                          'label': label,          
                          'value': pair} for label, pair in pairs['Spain'].items()],
            value = list(pairs['Spain'].values())[0],
            placeholder = 'Select a pair'
        ),
        
        html.Hr(),
        dcc.Markdown("""
             **Choose factor**
             """
        ),
        dcc.Dropdown(
            id = 'factor-selector',
            options = [{'label': i,          
                        'value': i} for i in FACTORS],
            value = FACTORS[0],
            placeholder = 'Select a factor'
        ),
        
        html.Hr(),
        dcc.Markdown("""
             **Choose plot-type**
             """
        ),
        dcc.Dropdown(
            id = 'plot-type',
            options = [{'label': i,          
                        'value': i} for i in ['normalized', 'spread', 'ratio']],
            value = 'normalized',
            placeholder = 'Select type'
        ),
            
    ],
    style = SIDEBAR_STYLE
    ),
                     
    html.Div([

            ### KPI - 1
            html.Div([
                dcc.Graph(id='1'),
                
                
                ], style = {'width': '33%',
                            # 'float': 'right',
                            'display': 'inline-block',
                            'background-color': '#f7f5fa'
                    }),
                            
            ### KPI - 2
            html.Div([
                dcc.Graph(id='2'),
                
                
                ], style = {'width': '33%',
                            # 'float': 'right',
                            'display': 'inline-block',
                            # 'background-color': '#f7f5fa'
                    }),
                            
            ### KPI - 3
            html.Div([
                dcc.Graph(id='3'),
                
                
                ], style = {'width': '33%',
                            # 'float': 'right',
                            'display': 'inline-block',
                            'background-color': '#f7f5fa'   
                    }),
                        
        # Main graphs
        
         html.Div([
             html.Div([
        
             dcc.Graph(id='pair-plot')
             ], style = {'width': '65%',
                         # 'float': 'right',
                         'display': 'inline-block',
                         }
                     ),
             html.Div([
             dcc.Graph(id='factor-correlation')
             ], style = {'width': '33%',
                         # 'float': 'right',
                         'display': 'inline-block',
                         }
                     )
                        
                 ])
                        
        ], style = {'width': '78%',
                    # 'height': '90%',
                    'float': 'right',
                    'display': 'inline-block'})
                        
])
          
                    
@app.callback(
    Output('pairs-selector', 'options'),
    Input('country-selector', 'value')
    )
def update_pairs(country):
    return [{'label': label,          
              'value': pair} for label, pair in pairs[country].items()]


@app.callback(
    Output('1', 'figure'),
    Input('pairs-selector', 'value')
    )
def update_KPI_1(p):
    
    p1 = p.split(' ')[0]
    p2 = p.split(' ')[1]
    
    pair = Pairs(p1, p2, START, END)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Indicator(
        mode   = 'number',
        value  = pair.coint(),
        title  = 'p-value of pair cointegration',
        ))

    return fig1


@app.callback(
    Output('2', 'figure'),
    Input('pairs-selector', 'value'),
    Input('factor-selector', 'value')
    )
def update_KPI_2(p, factor_name):
    
    p1 = p.split(' ')[0]
    p2 = p.split(' ')[1]
    
    pair = Pairs(p1, p2, START, END)
    
    s = pair.spread()
    factor = factor_frame[factor_name]
    
    s = s.reindex(factor.index, axis='index')
    
    # assert s.index == factor.index, "indexes not equal"
    assert len(s.index) == len(factor.index), "indexes not of equal length"
    
    corr = s.corr(factor)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Indicator(
        mode   = 'number',
        value  = corr,
        title  = 'correlation of spread with chosen factor',
        ))

    return fig2


@app.callback(
    Output('3', 'figure'),
    Input('pairs-selector', 'value')
    )
def update_KPI_3(p):
    fig3 = go.Figure()
    fig3.add_trace(go.Indicator(
        mode   = 'number',
        value  = 0,
        title  = 'empty for now',
        ))
    # fig3.update_layout(
    #     height=200,
    #     width=200
    #     )
    return fig3


@app.callback(
    Output('pair-plot', 'figure'),
    Input('pairs-selector', 'value'),
    Input('plot-type', 'value')
    )
def update_pair_plot(p, p_type):
    
    p1 = p.split(' ')[0]
    p2 = p.split(' ')[1]
    
    pair = Pairs(p1, p2, START, END)
    
    if p_type == 'normalized':

        return pair.plot_pair_norm(plotly_plot=(True))
    
    elif p_type == 'spread':
        
        return pair.spread(plot=True)
    
    else:
        
        return pair.ratio(plot=True)


@app.callback(
    Output('factor-correlation', 'figure'),
    Input('pairs-selector', 'value'),
    Input('factor-selector', 'value')
    )
def update_factor_correlation(p, factor_name):
    
    p1_ = p.split(' ')[0]
    p2_ = p.split(' ')[1]
    
    pair_ = Pairs(p1_, p2_, START, END)
    
    s_ = pair_.spread()
    factor_ = factor_frame[factor_name]
    
    s_ = s_.reindex(factor_.index, axis='index')
    
    # assert s.index == factor.index, "indexes not equal"
    assert len(s_.index) == len(factor_.index), "indexes not of equal length"
    
    fig5 = px.scatter(x = s_, y=factor_, trendline='ols')
    
    fig5.update_layout(title = 'Correlation with factor',
                   
                  yaxis_zeroline=False,
                  yaxis=dict(showgrid=False),
                  # yaxis_showticklabels = False,
                  
      
                  xaxis_zeroline=False,
                  xaxis=dict(showgrid=False),
                  # xaxis_showticklabels = False,
      
                    )
    
    return fig5
  


if __name__ in '__main__':
    app.run_server()
    

