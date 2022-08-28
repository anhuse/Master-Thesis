# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:42:22 2022

@author: Anders Huse
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objects as go

import scipy.stats as stats

import pandas as pd
import yfinance as yf


#%% Reading, preparing and defining data
df = pd.read_excel('currency1.xlsx', index_col='Date')
df.index = pd.to_datetime(df.index)
df = df.pct_change()[1:]

start_date = '2018-01-01'
end_date = df.index[-1]


companies = ['BN.PA', 'WCH.DE', 'LUNE.ST', 'XOM', 'AMZN', 'AAPL']

### Database for all the companies (may run for a while if many companies are included)
data_base = {}
corr_base = {}

for c in companies:
    
    ti = yf.Ticker(c)
    curr = ti.info['currency']
    close_price = ti.history(start=start_date, end=end_date)['Close']
    returns = close_price.pct_change()
    
    df_ = pd.merge(returns, df[curr], left_index=True, right_index=True)[1:]
    df_.columns = ['Return', curr]
    
    ### calculating correlation, here pearson correlation coefficient and p-value
    r, p = stats.pearsonr(df_.iloc[:,0], df_.iloc[:,1])
    
    corr_base[c] = (r, p)
    
    
    
    
    data_base[c] = df_

    
#%% Layout

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1('Company currency sensetivity'),
        dcc.Markdown("""
                     This dashborad calculates and visualizes the correlation between
                     the chosen company returns and the corresponding traded currency returns.
                     The correlation between the two return-series gives an indication
                     of how sensitive the company returns are to changes in the currency.
                     
                     This dashboard only examines the traded currency of the chosen company.
                     The company may be (is most likely) exposed to other currencies,
                     and so a closer examination of each companies business model
                     may be beneficiary to get a better view of the currencies at play.
                     """)],
                     style = {'width': '65%', 'display': 'inline-block'}
                ),
        html.Div([
            html.H3('Choose company'),
            dcc.Dropdown(id='company-selection',
                         options = [{
                            'label': '{}'.format(i),          
                            'value': i} for i in companies],
                        value = companies[0])
            ], style = {'width': '50%'}
                ),
        dcc.Graph(id='pearson-table'),
        dcc.Graph(id='currency-correlation')
        ])
    
    

@app.callback(
    Output('currency-correlation', 'figure'),
    Input('company-selection', 'value')
    )
def update_pairs_graph(company):
            
    ### plot
    fig = px.scatter(data_base[company], x=data_base[company].iloc[:,1],
                                         y=data_base[company].iloc[:,0],
                                         trendline='ols')
    
    fig.update_traces(marker_size=6.5,
                      )
    fig.update_layout(title=f'Regression of {company} over {data_base[company].columns[1]}',
                      yaxis=dict(title_text=company,
                            titlefont=dict(size=14)),
                      xaxis=dict(title_text=data_base[company].columns[1],
                            titlefont=dict(size=14)))
    return fig

@app.callback(
    Output('pearson-table', 'figure'),
    Input('company-selection', 'value'))

def update_table(company):
    
    fig = go.Figure(
        data = [go.Table(columnwidth = [20, 20],
                        header=dict(values=['correlation coefficient', 'p-value'],
                                    align=['left', 'left'],
                                     font_size=14),
                         cells=dict(values=[f'{corr_base[company][0]:.3f}',
                                           [corr_base[company][1]]],
                                    align=['left', 'left'],
                                    font_size=14,
                                    height=30))])
    fig.update_layout(height=220,
                      width=1000,
                      # paper_bgcolor="LightSteelBlue",
                      margin=dict(b=5),
                      title='Pearson correlation stats'
                      )
        
    return fig



                 
                 
                 
if __name__ in '__main__':
    app.run_server()