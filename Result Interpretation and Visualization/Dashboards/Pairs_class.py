# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:33:15 2022

@author: Anders Huse
"""


import pandas as pd
import yfinance as yf
from datetime import datetime

import plotly.graph_objects as go

import matplotlib.pyplot as plt

# import statsmodels.api as sm
# from statsmodels.tsa.stattools import coint
# 

class Pairs:
    
    def __init__(self,sec_1, sec_2, start, end, granularity='1d'):
        
        self.sec_1 : str = sec_1
        self.sec_2 : str = sec_2
        self.start : str = start
        self.end   : str   = end
        self.granularity : str = granularity
        
        df = yf.download(self.sec_1 + ' ' + self.sec_2,
                                  start = self.start,
                                  end   = self.end,
                                  interval= self.granularity)['Close']
        self.pair_df = df.fillna(method='bfill')
        
        assert self.pair_df.isnull().sum().sum() == 0, "null values exist"
        
        self.S1 = self.pair_df.iloc[:,0]
        self.S2 = self.pair_df.iloc[:,1]
        
        
        
        
    def plot_pair_norm(self, figsize=(12,6), plotly_plot=False):
        """Plots the normalized price series of the pair"""
        
        df_ = self.pair_df / self.pair_df.iloc[0] 
        
        if plotly_plot == True:
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = df_.index,
                                     y = df_.iloc[:,0],
                                     mode = 'lines',
                                     name =  self.sec_1)
                              )
            fig.add_trace(go.Scatter(x = df_.index,
                                     y = df_.iloc[:,1],
                                     mode = 'lines',
                                     name = self.sec_2)
                              )
            fig.update_layout(title = 'Normalized plot of the pair',
                               
                              yaxis_zeroline=False,
                              yaxis=dict(showgrid=False),
                              # yaxis_showticklabels = False,
                  
                              xaxis_zeroline=False,
                              xaxis=dict(showgrid=False),
                              # xaxis_showticklabels = False,
                  
                               )
            return fig
        
        else:
            
            fig = df_.plot(figsize=figsize, xlabel='')
            return fig

        
    
    def ratio(self, cutoff=1.6, plot=False, z=False):
        """Calculates the ratio of the pair. Returns plot if specified"""
        
        ratio = self.pair_df.iloc[:,0] / self.pair_df.iloc[:,1]
        ratio = ratio.fillna(method='bfill')
        ratio = (ratio - ratio.mean()) / ratio.std()
            
        if plot:
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = self.pair_df.index,
                                     y = ratio,
                                     mode = 'lines',
                                     name =  'spread')
                              )
            fig.add_hline(y = cutoff,
                          line_width = 1,
                          line_dash = 'dash',
                          line_color = 'red', row=2,col=1)
            
            fig.add_hline(y = -cutoff,
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
            return fig
    
    def spread(self, plot=False) -> pd.Series:
        """Calculates the spread of the pair. Returns plot if specified"""
        S1_ = self.pair_df.iloc[:,0]
        S2_ = self.pair_df.iloc[:,1]
    
    
        self.S1 = sm.add_constant(S1_)
        results = sm.OLS(self.S2, self.S1).fit()
        self.S1 = self.S1[self.sec_1.upper()]
        b = results.params[self.sec_1.upper()]
        spread = self.S2 - b * self.S1
        
        spread.index = spread.index.astype(str)
        
        if plot == True:
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = self.pair_df.index,
                                     y = spread,
                                     mode = 'lines',
                                     name =  'spread')
                              )
            fig.add_hline(y = spread.mean(),
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
    
        else:
        
            return spread

    
    
    def coint(self, return_score=False):
        """Calculates cointegration the pair"""
        
        result = coint(self.S1, self.S2)
        pvalue = result[1]
        score = result[0]
        
        if return_score:
            return pvalue, score
        else:
            return pvalue

    @classmethod
    def correlations(cls, use_spread=True):
        """
        Calculates the correlation of the pair price series (specified by user)
        and the given factor time-series
        """
        
        pass
        
    


if __name__ == '__main__':
    par = Pairs('elk.ol', 'wch.de', '2020-01-01', '2022-01-01')
    sr = par.spread()
    
