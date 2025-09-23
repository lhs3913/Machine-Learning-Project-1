# Authors: Matthew Ayala, Maanav Contractor, Alvin Liu, Luke Sims
#
# Model: Logistic Regression
# Dataset: Predict if the client will subscribe to a term deposit
# Training Dataset File: '/data/bank.csv'
# Testing Dataset File: '/data/bank-full.csv'
#
# Dataset Citation: 
# [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
# In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.
#
# Available at: [pdf] http://hdl.handle.net/1822/14838
#               [bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import autograd

df = pd.read_csv("data/training.csv", sep = ";")
features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 
            'month', 'campaign', 'pdays', 'previous', 'poutcome']

print(df.head())
print(df.info())
print(df.columns)
print(df['y'].value_counts())
