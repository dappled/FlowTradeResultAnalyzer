# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:47:42 2017

@author: dong5
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import glob
import pandas as pd
import numpy as np
import talib
from functools import reduce

def algo(file):
    data = pd.read_csv(file, header = None, names=['d','o','h','l','c','v'], parse_dates=['d']) #pd.read_csv(file, header = None, names=['d','o','h','l','c','v'], index_col = [0], parse_dates=True)
    
    datad = data['d']
    datac = data['c']
    datah = data['h']
    datal = data['l']
    datao = data['o']
    datav = data['v']
  
    cdiff = datac.diff()
    abscdiff = abs(cdiff)
    gtzerocdiff = np.maximum(cdiff, 0)
    
    #cdiffsma = talib.MA(cdiff, 5, 0);
    
    #cdiffemacmp = np.empty(len(cdiffsma)) *np.nan
    #cdiffemacmp[1] = cdiff[1]
    #for i in range(2, len(cdiffsma)):
    #    cdiffemacmp[i] = (cdiffemacmp[i-1] * 4 + cdiff[i])/5
    
    xperiod = 9
    x = gtzerocdiff.ewm(com = xperiod - 1, min_periods = xperiod, adjust = False,ignore_na=False).mean() / abscdiff.ewm(com = xperiod - 1, min_periods = xperiod, adjust = False, ignore_na=False).mean()
    
    xdiff =  x.diff() #np.concatenate(([np.nan], np.diff(x)))
    xg2 = (xdiff > 0) & (xdiff.shift(1) < 0) & (datao / datac.shift() > 1.002)
    
    trades_idx = xg2.index[xg2]
    tl = len(data)
    returns = []
    for i in range(0,len(trades_idx)):
        idx = trades_idx[i]
        iend = idx + 1
        while iend < tl and datav[iend] == 0:
            iend = iend + 1
        if iend < tl :
            returns.append((datad[idx], datac[iend] / datac[idx] - 1))
        
  
    return returns
#    datacnp = np.array(datac)  
#    yperiod = 3
#    y = ((2 * talib.MA(datacnp, yperiod, 2) + datah.rolling(window=yperiod,min_periods=yperiod,center=False).max() + datal.rolling(window=yperiod,min_periods=yperiod,center=False).min())/4).rolling(window=yperiod,min_periods=yperiod,center=False).mean()
#    
#    z = datac.pct_change() > 0.04 & datah/datac < 1.005
    
if __name__ == '__main__':
    allreturns = {}
    
    for fn in glob.glob(rf'D:\sdata\*.csv'):
        allreturns[fn.split('\\')[-1][:-4]] = algo(fn)
            
    input()
