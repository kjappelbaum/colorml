'''
Filename: /home/kevin/Dropbox (LSMO)/proj75_mofcolor/ml/notebooks/dump.py
Path: /home/kevin/Dropbox (LSMO)/proj75_mofcolor/ml/notebooks
Created Date: Wednesday, February 26th 2020, 1:38:43 pm
Author: Kevin Jablonka

Copyright (c) 2020 Your Company
'''
# This file contains some functions/code that I used at some points but no longer need at the same position but still want to keep track of 
augment_dict = {}

for color in list(survey['color_string'].unique()): 
    colors = survey[survey['color_string'] == color][['r', 'g', 'b']] 
    augment_dict[color] = colors.values

import pickle
with open('../data/augment_dict.pkl', 'wb') as fh: 
    pickle.dump(augment_dict, fh)
    
    