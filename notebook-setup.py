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

# INFO: to display dataframes as tables on call
from IPython.display import display


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

# INFO: Use logging like this:
logging.info("Logging is set!")


# INFO: Call for package/system spec output
# !%watermark --iversions

# %%

# INFO: Data spec (see cookie-cutter-data-science project)
DATA_DIR = "../../data"
DATA_QUALITY_DIR = "raw"
DATA_FILENAME = "dataset_1.parquet.gzip"

df_dev = pd.read_csv(f"{DATA_DIR}/{DATA_QUALITY_DIR}/{DATA_FILENAME}")
df_dev = pd.read_parquet(f"{DATA_DIR}/{DATA_QUALITY_DIR}/{DATA_FILENAME}")


# INFO: Typical data standartizarion, better perform withing the data loading cell

# INFO: Convert all columns to lowercase
df_dev.columns = df_dev.columns.str.lower()

# INFO: Parse datetime YYYY/MM/DD if unparsed
df_dev["deal_date"] = pd.to_datetime(df_dev["deal_date"], format="%Y-%m-%d")

# INFO: Cast string/object datatype on column with numeric input but string semantics
# (use string but not object for parquetization)
df_dev = df_dev.astype({"id": str})


# %%
# %%

# INFO: Round date to month end date (useful for downsampling daily deal data to monthly)
from pandas.tseries.offsets import MonthEnd

df_dev["deal_month"] = df_dev["deal_date"].transform(lambda x: MonthEnd(1).rollforward(x))


# INFO: Rollback date to previous month end (useful for monthly report data)
df_dev["last_month"] = pd.to_datetime(df_dev["deal_date"], format="%Y%m") + MonthEnd(-1)

# %%
# INFO: When you want to version your dataset
# via config or custom input for cleaning/testing/production needs.
# Requires config module
try:
    from config import DATASET_BUILD_VERSION
except Exception as e:
    print(e)
    print("using localy defined parameters...")
    # DATASET_BUILD_VERSION = 20201119
    DATASET_BUILD_VERSION = 20210129

# %%
# INFO: When you want to use fixed config/set of configs for ML model testing/production

from lightgbm import LGBMRegressor
from config import DEFAULT_MODEL_CONFIG

X = np.ones((1000, 4))
y = np.ones((1000, 1))

model = LGBMRegressor(**DEFAULT_MODEL_CONFIG).fit(X, y)
# %%

# INFO: for snapshot-testing Machine Learning model outputs
# Useful when refactoring ML code, to verify you didn't break stuff

result_1 = 5.977123123
result_2 = 1.123123

try:
    assert round(result_1, 3) == 5.977
    assert round(result_2, 3) == 5.587

except Exception as e:
    print("Oops!", e.__class__, "occurred.")
else:
    print("tests passed!")

# %%

# INFO: Modeling utilities


def _split_Xy(Xy, target: str = None):

    Xy = Xy
    X = Xy.drop(columns=target)
    y = Xy[target]

    return X, y


def _mape(y_true, y_pred):
    mape = np.mean(abs((y_true - y_pred) / (y_true + np.finfo(float).eps))) * 100
    return mape


def _smape(y_true, y_pred):
    smape = np.mean(
        (np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true) + np.finfo(float).eps))
    )  # .fillna(0))
    return smape


def _eval_regression_metrics(y_true, y_pred):
    """
    Pass series/array, not df

    use **kwargs to pass additional metrics TODO: implement kwargs
    """

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import median_absolute_error
    from sklearn.metrics import explained_variance_score

    evals = {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "MSE": round(mean_squared_error(y_true, y_pred), 3),
        "RMSE": round(mean_squared_error(y_true, y_pred, squared=False), 3),
        #              'MSLE': mean_squared_log_error(y_true, y_pred),
        "MAD": round(median_absolute_error(y_true, y_pred), 3),
        "Explained variation": round(explained_variance_score(y_true, y_pred), 3),
        "SMAPE": round(_smape(y_true, y_pred), 3),  # non applicable for negative/0/nan values
        "MAPE": round(_mape(y_true, y_pred), 3),
    }

    return evals


def _create_apply_transformers(df):
    from sklearn_pandas import DataFrameMapper
    import category_encoders as ce

    data_raw = df

    obj_cols = data_raw.select_dtypes("object").columns.to_list()

    from sklearn_pandas import gen_features

    feature_def = gen_features(
        columns=obj_cols,
        classes=[{"class": ce.OrdinalEncoder, "handle_unknown": "return_nan", "handle_missing": "return_nan"}],
    )

    mapper = DataFrameMapper(feature_def, default=None, df_out=True)

    data_transformed = mapper.fit_transform(data_raw)
    return data_transformed, mapper
