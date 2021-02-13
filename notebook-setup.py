# %%
# INFO: IPython extension for auto reloading modified custom packages"""
# !%load_ext autoreload
# !%autoreload 2

# INFO: IPython extension for package/system spec output
# !%load_ext watermark

# INFO: Core imports for practicaly any data science activity
import numpy as np
import pandas as pd

# INFO: Customize settings for Pandas
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
pd.options.display.max_colwidth = 500

# INFO: Plotting setup (matplotlib is only for compatibility with legacy code)
# import matplotlib.pyplot as plt
# !%matplotlib inline
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

# INFO: Customize plotting backend for Pandas (matplotlib for compat)
pd.options.plotting.backend = "plotly"
# pd.options.plotting.backend = "matplotlib"

# INFO: Customize Plotly theme
pio.templates.default = "plotly_dark"
# pio.templates.default = "plotly"

# INFO: Logging setup (replaces 'print' in development & seamlessly transitions to production code)
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# INFO: Call for package/system spec output
# !%watermark --iversions
