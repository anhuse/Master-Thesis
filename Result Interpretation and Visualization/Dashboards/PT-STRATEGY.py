# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:05:40 2022

@author: Anders Huse
"""



import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import pandas as pd
from datetime import datetime
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px

from datetime import date

import sys

sys.path.append("/Users/Anders Huse/Documents/Masteroppgave")
from functionality import TRADE, get_factors_or_indices, calc_spread,\
                          train_test_split, calc_correlations
                          
tt_split = 0.7


#%% Pairs

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

### -> foreign ticker always first

VALUTA = 'E'

pair_names = list(pairs['Germany']['Brenntag - Yara'].split(' '))
# pair_names = list(pairs['Sweden']['Lundin - Equinor'].split(' '))

foreign_ticker = pair_names[0]
obx_ticker = pair_names[1]

f_name = foreign_ticker.upper()
o_name = obx_ticker.upper()

START = '2013-01-01'
END = '2022-01-01'

### Getting the data

pair = yf.download(pair_names, start=START, end=END)['Close']
pair.index = pair.index.astype(str)
pair.fillna(method='ffill', inplace=True)

#%% valuta
df_forex = pd.read_excel('ValutaC.xlsx',
                         index_col='Date')

df_forex.rename(columns={ 'EUR': 'E',
                          'USD': 'U$',
                          'GBP': '£',
                          'CAD': 'C$',
                          'AUD': 'A$',
                          'JPY': 'Y',
                          'SEK': 'SK',
                          'DKK': 'DK',
                          'RUB': 'UR'}, inplace=True)
df_forex.index = df_forex.index.astype('str')

#%% Getting the valuta right for the foreign ticker

forex = df_forex[VALUTA]
forex = forex.reindex_like(pair)
forex = forex.fillna(method='bfill')

pair[f_name] = pair[f_name].mul(forex)

#%% Factors
factor_frame = get_factors_or_indices('data', 'Factors', start=START, end=END)

factor_names = {
    
    'Crude Oil-WTI Spot Cushing U$/BBL': 'Crude Oil WTI',
    'Europe Brent Spot FOB U$/BBL Daily': 'Brent Spot Europe',
    'LME-Copper Grade A Cash U$/MT': 'LME - Copper',
    'Baltic Exchange Dry Index (BDI) - PRICE INDEX': 'Baltic Dry (BDI)',
    'LME-Aluminium 99.7% Cash U$/MT': 'LME - Aluminium',
    'Gold Bullion LBM $/t oz DELAY': 'Gold',
    'Crude Oil BFO M1 Europe FOB $/BBl': 'Crude Oil Europe',
    'RFV Natural Gas TTF NL 1st Fut. Day - SETT. PRICE': 'RFV Natural Gas1',
    'ICE Natural Gas 1 Mth.Fwd. P/Therm': 'ICE Natural Gas',
    'RFV Natural Gas TTF NL 1st Fut. Mth - SETT. PRICE': 'RFV Natural Gas2',
    'Fish Pool Salmon TRc1 NOK/KG - SETT. PRICE': 'Fish Pool Salmon',
    'LME-Nickel Cash U$/MT': 'LME - Nickel',
    'LME-Cobalt 3 Month': 'LME-Cobalt',
    'Raw Sugar-ISA Daily Price c/lb': 'Raw Sugar',
    'Soyabeans, No.1 Yellow $/Bushel': 'Soybeans'
}

factor_frame.rename(columns=factor_names, inplace=True)

### Dropping columns

drop_columns = ['LME - Copper', 'ICE Natural Gas', 'RFV Natural Gas2', 'Salmon Frozen Export',
               'Salmon Fresh or Chilled Export', 'Fish Pool Salmon', 'Fish Pool Index Spot Salmon NOK/KG',
               'LME - Nickel', 'LME-Cobalt', 'Raw Sugar', 'Soybeans', 'NYMEX Natural Gas Henry Hub C1']

factor_frame.drop(drop_columns, axis=1, inplace=True)
factor_frame = factor_frame.astype('float')

### null-values

factor_frame.isnull().sum()
factor_frame.dropna(axis='columns', inplace=True)

assert factor_frame.isnull().sum().sum() == 0, "Null values present in factors"

#%% Ratio / Spread

spread = calc_spread(pair[f_name],
                     pair[o_name],
                     [name for name in pair.columns if not name.endswith('.OL')][0],
                     [name for name in pair.columns if name.endswith('.OL')][0])
spread = spread.fillna(method='bfill')
z_score_spread = (spread - spread.mean()) / spread.std()

ratio = pair[f_name] / pair[o_name]
ratio = ratio.fillna(method='bfill')
z_score_ratio = (ratio - ratio.mean()) / ratio.std()

#%% STRATEGY

train, test, split = train_test_split(pair, tt_split)

f_trade = test[f_name]
o_trade = test[o_name]

strategy_profit = TRADE(f_trade, o_trade,
                        short_window = 12,
                        long_window = 80,
                        cutoff_inner = 0.33,
                        cutoff_outer = 1.2,
                        rolling_z=False,
                        print_actions=False)
print(f'Profit from strategy: {strategy_profit:,.2f}')

#%% Correlations
corr = calc_correlations(ratio, factor_frame_pct)

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

#%% App layout

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
        
    html.Div(
    [
    
        html.H1("Strategy Validation"),
        html.Hr(),
        html.P(
            "Dashboard for examination of the implementation of the pairs\
                trading strategy."
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
             **Date Range**
             """
        ),
        dcc.DatePickerRange(
            id = 'date-range-selector',
            min_date_allowed = date(2012, 1, 1),
            max_date_allowed = date(2021, 1, 1),
            display_format = 'YYYY-MM-DD',
            initial_visible_month = date(2013,1,1),
            end_date = date(2021, 1, 1),
            start_date = date(2012, 1, 1)

        ),
        
        html.Hr(),
        dcc.Markdown("""
             **Train / test split**
             """
        ),
        dcc.Input(
            id = 'tt-split',
            type = 'number',
            min = 0.5,
            max = 0.9,
            step = 0.05,
            placeholder = '',
            value = 0.7,
            style = {'marginRight': '1px'}
        ),
        
        html.Hr(),
        dcc.Markdown("""
             **Timeseries generating the signals**
             """
        ),
        dcc.RadioItems(
                id = 'time-series-selector',
                options = [
                        {'label': 'ratio', 'value': 'ratio'},
                        {'label': 'spread', 'value': 'spread'}
                        ],
                inputStyle = {'margin-right': '5px',
                               # 'margin-top': '20px'
                              },
                labelStyle={'display': "block"},
                value = 'ratio'),


        html.Hr(),
        dcc.Markdown("""
             **Rolling / daily stats**
             """
        ),
        dcc.RadioItems(
                id = 'rolling-daily',
                options = [
                        {'label': 'rolling', 'value': 'rolling'},
                        {'label': 'daily', 'value': 'daily'}
                        ],
                inputStyle = {'margin-right': '5px',
                               # 'margin-top': '20px'
                              },
                labelStyle={'display': "block"},
                value = 'daily'),
            
    ],
    style = SIDEBAR_STYLE
    ),
                     
    html.Div([

            ### KPI - 1
            html.Div([
                dcc.Graph(id='1'),
                
                
                ], style = {'width': '25%',
                            # 'float': 'right',
                            'display': 'inline-block',
                            'background-color': '#f7f5fa'
                    }),
                            
            ### KPI - 2
            html.Div([
                dcc.Graph(id='2'),
                
                
                ], style = {'width': '25%',
                            # 'float': 'right',
                            'display': 'inline-block',
                            # 'background-color': '#f7f5fa'
                    }),
                            
            ### KPI - 3
            html.Div([
                dcc.Graph(id='3'),
                
                
                ], style = {'width': '25%',
                            # 'float': 'right',
                            'display': 'inline-block',
                            'background-color': '#f7f5fa'   
                    }),
                            
            ### KPI - 4
            html.Div([
                dcc.Graph(id='4'),
                
                
                ], style = {'width': '25%',
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
    Output('pair-plot', 'figure'),
    Input('pairs-selector', 'value'),
    Input('date-range-selector', 'start_date'),
    Input('date-range-selector', 'end_date'),
    Input('tt-split', 'value'),
    Input('time-series-selector', 'value'),
    Input('rolling-daily', 'value')
    )
def update_pair_graph(pair_names, start_date, end_date,\
                      tt_split, ratio_spread, rolling_daily):
    
    foreign_ticker = pair_names.split(' ')[0]
    obx_ticker = pair_names.split(' ')[1]

    f_name = foreign_ticker.upper()
    o_name = obx_ticker.upper()

    ### Getting the data
    
    pair = yf.download(pair_names, start=start_date, end=end_date)['Close']
    pair.index = pair.index.astype(str)
    pair.fillna(method='ffill', inplace=True)
    
    ### Ratio / Spread

    spread = calc_spread(pair[f_name],
                         pair[o_name],
                         [name for name in pair.columns if not name.endswith('.OL')][0],
                         [name for name in pair.columns if name.endswith('.OL')][0])
    spread = spread.fillna(method='bfill')
    z_score_spread = (spread - spread.mean()) / spread.std()
    
    ratio = pair[f_name] / pair[o_name]
    ratio = ratio.fillna(method='bfill')
    z_score_ratio = (ratio - ratio.mean()) / ratio.std()
    
    if ratio_spread == 'ratio':
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = pair.index,
                                 y = z_score_ratio,
                                 mode = 'lines',
                                 name =  'spread')
                          )
        fig.add_hline(y = 0,
                      line_width = 1,
                      line_dash = 'dash',
                      line_color = 'black', row=2,col=1)
        fig.add_hline(y = 2,
                      line_width = 1,
                      line_dash = 'dash',
                      line_color = 'red', row=2,col=1)
        
        fig.add_hline(y = -2,
                      line_width = 1,
                      line_dash = 'dash',
                      line_color = 'green', row=2,col=1)            
    
        fig.update_layout(title = 'Ratio of the pair',
                           
                          yaxis_zeroline=False,
                          yaxis=dict(showgrid=False),
                          # yaxis_showticklabels = False,
              
                          xaxis_zeroline=False,
                          xaxis=dict(showgrid=False),
                          # xaxis_showticklabels = False,
              
                           )        
        
        
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = pair.index,
                                 y = z_score_spread,
                                 mode = 'lines',
                                 name =  'spread')
                          )
        fig.add_hline(y = z_score_spread.mean(),
                      line_width = 1,
                      line_dash = 'dash',
                      line_color = 'black', row=2,col=1)

        fig.update_layout(title = 'Spread of the pair',
                           
                          yaxis_zeroline=False,
                          yaxis=dict(showgrid=False),
                          # yaxis_showticklabels = False,
              
                          xaxis_zeroline=False,
                          xaxis=dict(showgrid=False),
                          # xaxis_showticklabels = False,
                           )
    return fig
     
@app.callback(
    Output('1', 'figure'),
    Input('pairs-selector', 'value'),
    Input('date-range-selector', 'start_date'),
    Input('date-range-selector', 'end_date'),
    Input('tt-split', 'value'),
    Input('time-series-selector', 'value'),
    Input('rolling-daily', 'value')
    )
def update_KPI_1(pair_names, start_date, end_date,\
                      tt_split, ratio_spread, rolling_daily):
    
    foreign_ticker = pair_names.split(' ')[0]
    obx_ticker = pair_names.split(' ')[1]

    f_name = foreign_ticker.upper()
    o_name = obx_ticker.upper()

    ### Getting the data
    
    pair = yf.download(pair_names, start=start_date, end=end_date)['Close']
    pair.index = pair.index.astype(str)
    pair.fillna(method='ffill', inplace=True)
    
    ### Ratio / Spread

    spread = calc_spread(pair[f_name],
                         pair[o_name],
                         [name for name in pair.columns if not name.endswith('.OL')][0],
                         [name for name in pair.columns if name.endswith('.OL')][0])
    spread = spread.fillna(method='bfill')
    z_score_spread = (spread - spread.mean()) / spread.std()
    
    ratio = pair[f_name] / pair[o_name]
    ratio = ratio.fillna(method='bfill')
    z_score_ratio = (ratio - ratio.mean()) / ratio.std()
    
    ### Valuta 
    
    forex_ = df_forex[VALUTA]
    forex_ = forex.reindex_like(pair)
    forex_ = forex.fillna(method='bfill')
    
    pair[f_name] = pair[f_name].mul(forex)
    
    train, test, split = train_test_split(pair, tt_split)

    f_trade = test[f_name]
    o_trade = test[o_name]
    
    rolling = True if rolling_daily == 'rolling' else False
    
    strategy_profit = TRADE(f_trade, o_trade,
                            short_window = 12,
                            long_window = 80,
                            cutoff_inner = 0.33,
                            cutoff_outer = 1.2,
                            rolling_z=rolling,
                            print_actions=False)
    print(f'Profit from strategy: {strategy_profit:,.2f}')
    
    fig1 = go.Figure()
    fig1.add_trace(go.Indicator(
        mode   = 'number',
        value  = strategy_profit,
        title  = 'Profit from strategy',
        ))

    return fig1



    return fig
    

    

    
    # ## Factors 
    # factor_frame_ = factor_frame.reindex(pair.index, axis='index')
    # factor_frame_['forex'] = forex
    
    # ### Writing to percentage change
    
    # factor_frame_ = factor_frame_.pct_change()[1:]
    # factor_frame_.fillna(method='bfill', inplace=True)
    
    # assert factor_frame_.isnull().sum().sum() == 0, "null-values present"
    
    # ### Ratio / Spread

    # spread = calc_spread(pair[f_name],
    #                      pair[o_name],
    #                      [name for name in pair.columns if not name.endswith('.OL')][0],
    #                      [name for name in pair.columns if name.endswith('.OL')][0])
    # spread = spread.fillna(method='bfill')
    # z_score_spread = (spread - spread.mean()) / spread.std()
    
    # ratio = pair[f_name] / pair[o_name]
    # ratio = ratio.fillna(method='bfill')
    # z_score_ratio = (ratio - ratio.mean()) / ratio.std()

    # ### STRATEGY

    # train, test, split = train_test_split(pair, tt_split)
    
    # f_trade = test[f_name]
    # o_trade = test[o_name]
    
    # strategy_profit = TRADE(f_trade, o_trade,
    #                         short_window = 12,
    #                         long_window = 80,
    #                         cutoff_inner = 0.33,
    #                         cutoff_outer = 1.2,
    #                         rolling_z=False,
    #                         print_actions=False)


    ### Correlations
    # corr = calc_correlations(ratio, factor_frame_pct)
        



    
    
               
                    
                    

                    
                    
                    
                    
                    
                    
if __name__ in '__main__':
    app.run_server()
    