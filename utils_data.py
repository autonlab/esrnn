import numpy as np
import pandas as pd

import torch

def get_m4_all_series(data_dir='/data/', dataset_name, max_series = 100, data='train'):
    m4_info = pd.read_csv(data_dir+'M4-info.csv', usecols=['M4id','category'])

    if dataset_name == 'm4hourly':
      dataset = pd.read_csv(mc.data_dir+data+'/Hourly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('H')].reset_index(drop=True)
    elif dataset_name == 'm4daily':
      dataset = pd.read_csv(mc.data_dir+data+'/Daily-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('D')].reset_index(drop=True)
    elif dataset_name == 'm4weekly':
      dataset = pd.read_csv(mc.data_dir+data+'/Weekly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('W')].reset_index(drop=True)
    elif dataset_name == 'm4monthly':
      dataset = pd.read_csv(mc.data_dir+data+'/Monthly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('M')].reset_index(drop=True)
    elif dataset_name == 'm4quarterly':
      dataset = pd.read_csv(mc.data_dir+data+'/Quarterly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('Q')].reset_index(drop=True)
    elif dataset_name == 'm4yearly':
      dataset = pd.read_csv(mc.data_dir+data+'/Yearly-{}.csv'.format(data))
      m4_info = m4_info[m4_info['M4id'].str.startswith('Q')].reset_index(drop=True)

    all_series = []
    if max_series==-1:
      max_series = len(dataset)
    else:
      max_series = min(len(dataset), max_series)
    
    for i in range(max_series):
        row = dataset.loc[i]
        row = row[row.notnull()]
        category = m4_info.loc[i,'category']
        # omit row with column names
        y = row[1:].values
        #idx = row[0]
        m4_object = M4TS(mc, category, y, i)
        all_series.append(m4_object)
        
    return all_series
