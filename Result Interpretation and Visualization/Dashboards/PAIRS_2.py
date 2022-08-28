# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:08:55 2022

@author: Anders Huse
"""

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import pandas as pd
import numpy as np
import yfinance as yf
from progress.bar import Bar

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

import sys
sys.path.append("/Users/Anders Huse/Documents/Masteroppgave")

import functionality

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

CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem"
    }

#%% Global variables


# COUNTRIES = ['France', 'USA', 'UK', 'Germany',  'Spain', 'Italy',
#                        'Canada', 'Australia', 'Hong Kong', 'Japan', 'Swiss',
#                        'Russia', 'Sweden', 'Denmark', 'Finland']

COUNTRIES_all = ['Norway', 'France', 'US', 'UK', 'Italy', 'Canada',
             'Australia', 'Austria', 'Hong Kong', 'Japan', 'Sweden', 'Chile',
             'Denmark', 'Finland', 'China', 'US', 'UK', 'Spain', 'Germany', 'Swiss']
COUNTRIES_for = ['France', 'Italy', 'Canada',
             'Australia', 'Austria', 'Hong Kong', 'Japan', 'Sweden', 'Chile',
             'Denmark', 'Finland', 'China']

COUNTRIES_for_2 = ['US', 'UK', 'Spain', 'Germany', 'Swiss']

COUNTRIES_for_all = COUNTRIES_for + COUNTRIES_for_2

# OBX_tickers = ['ADE', 'AKER', 'AKRBP', 'KAHOT', 'BAKKA', 'DNB', 'EQNR', 'FRO',
#                'GJF', 'GOGL', 'MOWI', 'MPCC', 'NEL', 'NOD', 'NHY',
#                'ORK', 'RECSI', 'SALM', 'SCATC', 'SCHA', 'STB', 'SUBC', 'TEL',
#                'TOM', 'YAR']

# additional_tickers_NO_large_cap = [
#     'AFG', 'AKSO', 'AFK', 'ATEA', 'AUSS', 'AUTO', 'BONHR',
#     'BRG', 'CRAYN', 'DNO', 'ELK', 'ENTRA', 'EPR', 'KOG', 'LSG', 'NTS', 'OLT',
#     'SBANK', 'SCATC', 'SCHB', 'MING', 'SRBNK', 'TIETO', 'VEI', 'WAWI'
    
#     ]

# OBX_tickers.extend(additional_tickers_NO_large_cap)

# currency_mapping = {'France': 'EUR',
#                     'US': 'USD',
#                     'UK': 'GBP',
#                     'Italy': 'EUR',
#                     'Canada': 'CAD',
#                     'Australia': 'AUD', 
#                     'Austria': 'EUR',
#                     'Hong Kong': ,
#                     'Japan': ,
#                     'Sweden': ,
#                     'Chile': ,
#                     'Denmark': ,
#                     'Finland': ,
#                     'China': }

START = '2013-01-02'
END = '2019-01-01'
CLUSTERING_MODEL = 'DBSCAN'

DATA_fn = 'MASTER_A.xlsx'
VALUTA_fn = 'VALUTA_CLEAN.xlsx'

N_PRIN_COMPONENTS = 15

#%% Reading the data

data_all = {}
for country in COUNTRIES_for:
    data_all[country] = pd.read_excel('CLEANED_DATA.xlsx',
                                       sheet_name=country,
                                       index_col='Name')
    print(country + ' read')
    
for country in COUNTRIES_for_2:
    data_all[country] = pd.read_excel('CLEANED_DATA_2.xlsx',
                                       sheet_name=country,
                                       index_col='Name')[:END]
    print(country + ' read')
#%%
df_obx2 = pd.read_excel('CLEANED_DATA_2.xlsx',
                        sheet_name='NO_EQ_200',
                        index_col='Name')[:'2022-03-01']
df_obx2.drop('SEABIRD EXPLORATION', axis=1, inplace=True)

df_obx2.index = pd.to_datetime(df_obx2.index)
df_obx2 = df_obx2.resample('W').sum()

#%% Resampling to weekly observations
for country in COUNTRIES_for_all:
    data_all[country].index = pd.to_datetime(data_all[country].index)
    data_all[country] = data_all[country].resample('W').sum()

#%% Reading equity data

# data_equity_all = functionality.get_eqt_mult(COUNTRIES_all,
#                                              start=START,
#                                              filename=DATA_fn)

# df_obx = data_equity_all['Norway']
# data_equity_all.pop('Norway')

#%% Preparing data
# curr_mapping = functionality.curr_mapping(DATA_fn, COUNTRIES_for)
# unique_currencies = set(val for dic in curr_mapping.values() for val in dic.values())

#%% Foreign exchange currencires

# df_forex = pd.read_excel(VALUTA_fn, index_col='Unnamed: 0'
#                          )
# df_forex.rename(columns={'EUR': 'E',
#                          'USD': 'U$',
#                          'GBP': '£',
#                          'CAD': 'C$',
#                          'AUD': 'A$',
#                          'JPY': 'Y',
#                          'SEK': 'SK',
#                          'DKK': 'DK',
#                          'RUB': 'UR',
#                          'HKD': 'K$',
#                          'CNY': 'CH'
#                          }, inplace=True)
# df_forex = df_forex[START:END]
# df_forex.index = df_forex.index.astype(str)
# df_forex = df_forex.reindex(df_obx.index, axis='index')
# df_forex = df_forex.fillna(method='bfill', limit=3)

#%% mapping currencies to equities
# for country in data_equity_all.keys():
#     for equity, currency in zip(data_equity_all[country], curr_mapping[country].values()):
#         if currency in ['NK', 'CE']:
#             continue

#         data_equity_all[country][equity] =\
#         data_equity_all[country][equity].mul(df_forex[currency])
#         print(equity)
#     print('***********************', country)

# am = data_equity_all['US']
#%% OBX

# df_obx = yf.download([s + '.ol' for s in OBX_tickers],
#                  start = START,
#                  end   = END)['Adj Close']

# column_names= [t[:-3] for t in df_obx.columns]
# df_obx.columns = column_names
# df_obx.index = df_obx.index.astype(str)
# df_obx.dropna(axis=1, how='any', inplace=True)
# df_obx.tail()



#%% Foreign exchanges

# all = {}

# for c in COUNTRIES:
#     df_ = pd.read_excel('DATA/' + c + '.xlsx', index_col='Name')

#     assert df_.isnull().sum().sum() == 0, f"Null values present in {c}"

#     df_ = df_.reindex(df_obx.index, axis='index')
#     all[c] = df_

#%% Reindexing currencies

# df_forex = df_forex.reindex(df_obx.index, axis='index')
# df_forex.fillna(method = 'ffill')

#%%  Obtaining the currency mapping

# curr_mapping = functionality.curr_mapping_country(DATA_fn, COUNTRIES)

#%% Dashboard layout

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
        
    html.Div(
    [
    
        html.H3("Options"),
        html.Hr(),
        # html.P(
        #     "Setting optins for exploring possible pairs between the OBX and\
        #      a variety of foreign exchanges",
        #     ),
        dcc.Markdown("""**Display labels:** """),
        dcc.RadioItems(
            id = 'labels-check',
            options = [
                    {'label': 'True', 'value': 'True'},
                    {'label': 'False', 'value': 'False'}
                    ],
            inputStyle = {'margin-right': '5px',
                          'margin-bottom': '20px'},
            labelStyle={'display': "inline-block"},
            value = 'False'
            ),
        
        
        dcc.Markdown("""**Display noise-points:** """),
        dcc.RadioItems(
                id = 'noise-points',
                options = [
                        {'label': 'True', 'value': 'True'},
                        {'label': 'False', 'value': 'False'}
                        ],
                inputStyle = {'margin-right': '5px',
                               # 'margin-top': '20px'
                              },
                labelStyle={'display': "inline-block"},
                value = 'False'
                ),
            
        
        html.Hr(),
        dcc.Markdown("""
                     **Choose foreign exchange to examine**
                     """),
        dcc.Dropdown(
            id = 'country-selector',
            options = [{
                          'label': i,          
                          'value': i} for i in COUNTRIES_for_all],
            value = COUNTRIES_for_all[0]
        ),
        html.Hr(),
        dcc.Markdown("""
                     **Clustering algorithm**
                     """
            ),
        dcc.RadioItems(
            id = 'clustering-algo',
            options = [
                {'label': 'DBSCAN', 'value': 'DBSCAN'},
                {'label': 'OPTICS', 'value': 'OPTICS'},
                ],
            value='DBSCAN',
            labelStyle={'display': "block"} # alt. "inline-block"
            ),
        
        html.Hr(),
        
        dcc.Markdown("""
                     **DBSCAN Parameters:**
                     """),
        html.P('eps'),
        dcc.Input(
        id = 'eps',
        type = 'number',
        placeholder = 'eps: ',
        step = 0.1,
        value=1.2
        ),
        
        html.P('min-samples'),
        dcc.Input(
        id = 'min-samples-db',
        type = 'number',
        placeholder = 'min-sampels: ',
        value=2
        ),
        
        html.Hr(),
        
        dcc.Markdown("""
             **OPTICS Parameters:**
             """),
        
        dcc.Input(
        id = 'xi',
        type = 'number',
        placeholder = 'xi: ',
        step = 0.01,
        value=0.05
        ),
        html.P('xi (default: 0.05)'),
        
        
        dcc.Input(
        id = 'min-samples-op',
        type = 'number',
        placeholder = 'min-sampels: ',
        value=2
        ),
        html.P('min-samples (default: 5)'),
        
        
        dcc.Input(
        id = 'p',
        type = 'number',
        placeholder = 'p: ',
        step = 0.1,
        value=2
        ),
        html.P('p (default: 2)'),
        
        
            
    ],
    style = SIDEBAR_STYLE
    ),
    html.Div([
        html.H2("Pairs Exploration"),
        dcc.Graph(id='pairs-graph'),
        # dcc.Markdown("""
        #              Description of the graph:
        #              """)
        ], style = {'width': '78%', 'float': 'right',
                    'display': 'inline-block',
                    "padding": "2rem 1rem"
                    # "background-color": "#f7f5fa"
                    })
])
                    
                    
@app.callback(
    Output('pairs-graph', 'figure'),
    Input('country-selector', 'value'),
    Input('clustering-algo', 'value'),
    Input('eps', 'value'),
    Input('min-samples-db', 'value'),
    Input('xi', 'value'),
    Input('min-samples-op', 'value'),
    Input('p', 'value'),
    Input('labels-check', 'value'),
    Input('noise-points', 'value')
    )
def update_pairs_graph(COUNTRY, MODEL, EPS, MIN_DB, XI, MIN_OP, P, LABELS, NOISE):
    
    c = COUNTRY

    # curr = curr_mapping[c]   # currency symbol
    
    # forex = df_forex[curr]     # currency time series
    
    df_foreign = data_all[c]
    df_foreign = df_foreign.reindex(df_obx2.index, axis='index')
    print(df_foreign.shape)
    # df_foreign.apply(lambda x : x * forex, axis=0)  # Converting to NOK
    
    returns = pd.concat([df_obx2, df_foreign], axis=1).pct_change()[1:]
    print(returns.isna().any().sum())
    ### PCA
    pca = PCA(n_components=N_PRIN_COMPONENTS)
    pca.fit(returns)
    
    ### Scaling
    X = pca.components_.T
    X = preprocessing.StandardScaler().fit_transform(X)
    
    ### DBSCAN
    clf = DBSCAN(eps=EPS, min_samples=MIN_DB)
    clf.fit(X)
    
    labels_dbscan = clf.labels_
    
    ### OPTICS     
    clust = OPTICS(min_samples      = MIN_OP,
                   xi               = XI,
                   p                = P
                  )
    clust.fit(X)
    
    reachability = clust.reachability_[clust.ordering_]    
    labels_optics = clust.labels_[clust.ordering_]
    
    ### Setting the labels -> DBASCAN / OPTICS
    if MODEL == 'DBSCAN':
        labels = labels_dbscan
    else:
        labels = labels_optics
        
        
    label_series_all = pd.Series(index = returns.columns,
                         data = labels.flatten())
    label_series = label_series_all[label_series_all != -1]
    counts = label_series.value_counts()
    
    
    ### TSNE
    tsne = TSNE(learning_rate=10, perplexity=25, random_state=123)
    X_tsne = tsne.fit_transform(X)
    
    ### Plotting data
    
    ### DBSCAN
    
    if MODEL == 'DBSCAN':
        fig = go.Figure()
        clusters = {}
        
        for clust in set(labels[labels!=-1]):
            
            if LABELS == 'True':
                mode = 'markers+text'
                text = label_series[(label_series == clust)].index
            else:
                mode = 'markers'
                text = None
                
        
            fig.add_trace(go.Scatter(
                                x = X_tsne[(labels==clust), 0],
                                y = X_tsne[(labels==clust), 1],
                                mode = mode,
                                text = text,
                                marker = dict(color=clust,
                                              size=10),
                                name = f'C{clust} -> {len(X_tsne[(labels==clust)])} points',
                                opacity = 1
            ))
            

        ### Noise points
        
        if NOISE == 'True':
        
            fig.add_trace(go.Scatter(
                                x = X_tsne[(label_series_all ==-1), 0],
                                y = X_tsne[(label_series_all ==-1), 1],
                                mode = 'markers',
                                marker = dict(color='blue'),
                                name = 'non-cluster',
                                opacity = 0.1
            ))
        else:
            pass

        fig.update_layout(
                           title = f'Automatic Clustering {MODEL}',
                          
                          width=1200,
                          height=900,
                          
                          yaxis_zeroline=False,
                          yaxis=dict(showgrid=False),
        #                   yaxis_showticklabels = False,
                          
                          xaxis_zeroline=False,
                          xaxis=dict(showgrid=False),
        #                   xaxis_showticklabels = False,
                          
                          legend_title_text = 'Clusters:'
                          
                              # updatemenus = [
                
                              #   dict(buttons=list(buttons.values()))
                              # ]
                         )
    else:
        
        ### OPTICS
        space = np.arange(len(X))
        colors = ['red', 'blue', 'green', 'pink', 'purple', 'grey']
        
        fig = make_subplots(rows=2,
                            cols=1,
                            row_heights=[0.8,0.2],
                            vertical_spacing=0.1,
                            row_titles=['', 'Reachability']
                           )
        

        for clust, c in zip(set(labels[labels!=-1]), colors):
            
            if LABELS == 'True':
                mode = 'markers+text'
                text = label_series[(label_series == clust)].index
            else:
                mode = 'markers'
                text = None
                
            fig.add_trace(go.Scatter(
                                x = X_tsne[(labels==clust), 0],
                                y = X_tsne[(labels==clust), 1],
                                mode = mode,
                                text = text,
                                marker = dict(color=c,
                                              size=10),
        #                         name = f'C{clust} -> {len(X_tsne[(labels==clust)])} points',
                                opacity = 1
            ), row=1,col=1)
            
        if NOISE == 'True':
        
            fig.add_trace(go.Scatter(
                                x = X_tsne[(label_series_all ==-1), 0],
                                y = X_tsne[(label_series_all ==-1), 1],
                                mode = 'markers',
                                marker = dict(color='blue'),
                                name = 'non-cluster',
                                opacity = 0.1
                                ))
            
        for klass, c in zip(range(len(counts)), colors):
            Xk = space[labels == klass]
            Rk = reachability[labels == klass]
            
            fig.add_trace(
                go.Scatter(
                x = Xk,
                y = Rk,
                mode = 'markers',
                name= klass,
                marker = dict(color=c,size=8),
            ), row=2, col=1)
            
        fig.add_hline(y=0.5,
                      line_width = 1,
                      line_dash = 'dashdot',
                      line_color = 'black', row=2,col=1)
        fig.add_hline(y=2.0,
                      line_width = 1,
                      line_dash = 'dash',
                      line_color = 'black', row=2,col=1)
        
        
        fig.update_layout(title = 'OPTICS',
                          
                          width=1200,
                          height=800,
                          

                          
                          yaxis_zeroline=False,
                          yaxis=dict(showgrid=False),
        #                   yaxis_showticklabels = False,
                          
                          xaxis_zeroline=False,
                          xaxis=dict(showgrid=False),
        #                   xaxis_showticklabels = False,
                          
                          legend_title_text = 'Clusters:',
        #                   showlegend=False
                          
        #                       updatemenus = [
                
        #                         dict(buttons=list(buttons.values()))
        #                       ]
                         )
        
        fig.update_yaxes(title_text='hell', row=1, col=2)

        
    return fig




if __name__ in '__main__':
    app.run_server()
    
    
    