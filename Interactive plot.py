#!/usr/bin/env python
# coding: utf-8

# In[6]:


import plotly.express as px
import pandas as pd

data = pd.read_csv('Langevin_1D.csv', header=None)

fig = px.line(data, title='1D Langevin Trajectory')
fig.write_html('first_figure.html', auto_open=True)

