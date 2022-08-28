# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:37:34 2022

@author: Anders Huse
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/Anders Huse/Documents/Masteroppgave/Finding_pairs/')

import new
new.function(1, 2)


#%% This works
import os
os.chdir('/Users/Anders Huse/Documents/Masteroppgave/Finding_pairs/')
f = open('France.xlsx', 'r')


#%% This works
import sys
sys.path.append("/Users/Anders Huse/Documents/Masteroppgave/Finding_pairs/")
try:
    f = open('France.xlsx', 'r')
except:
    print('this did not work') # this will print
    
#%%
sys.path.insert(1, '/Users/Anders Huse/Documents/Masteroppgave/Finding_pairs/')

from xlrd import open_workbook

wb = open_workbook('dataC.xlsx')
#%%
pd.read_excel('dataC.xlsx')

#%%
import os
# sys.path.insert(1, '/Users/Anders Huse/Documents/Masteroppgave/Finding_pairs/')
sys.path.append("/Users/Anders Huse/Documents/Masteroppgave/Finding_pairs/")
import new
print(os.getcwd())
print(new.function(1, 2))

#%%
sys.path.insert(1, '/Users/Anders Huse/Documents/Masteroppgave/Master dashboards/DATA')
# a = pd.read_excel('DATA/' + 'Canada.xlsx')
b = pd.read_excel('DATA/Canada.xlsx')