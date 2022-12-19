"""
===================================
  Stock Analysis and Prediction 
===================================
"""
# Basic
from prophet import Prophet
import pandas_datareader as pdr
from matplotlib import pyplot as plt
import matplotlib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime as dtt
import time
import warnings  # remove warnings
warnings.filterwarnings('ignore')

# Modules

# Plot
# from matplotlib.ticker import FuncFormatter
# import seaborn as sns
# import pickle

# Stocker: A Stock Analysis and Prediction Toolkit using Additive Models
# from modules.stocker import Stocker
# from stocker import Stocker

# get stock data

# stock forecast model


# plot
# from streamlit_echarts import st_echarts
# import plotly.figure_factory as ff

st.set_page_config(layout="wide")

### Data Import ###
com_no = ''
startTime = ''
endTime = ''
filename = '2330_20000101_20221218.csv'

# @st.cache


def get_obs_data():
    """note"""
    tmp_df_database = pd.read_csv(f"./01_obs/{filename}")

    return tmp_df_database

# @st.cache


def get_ft_data():
    """note"""
    tmp_df_database = pd.read_csv(f"./02_ft/{filename}")

    return tmp_df_database


def show_data(data_x, data_y):
    """note"""

    option = {
        "xAxis": {
            "type": "category",
            "data": [],
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "data": [],
                "type": "line"
            }
        ],
    }

    option['xAxis']['data'] = data_x
    option['series']['data'] = data_y

    st_echarts(
        options=option, height="600px",
    )


def stock_analytic(df_stock):
    """"""
    # 定義模型 Facebook
    model = Prophet()

    # 訓練模型
    model.fit(df_stock)

    # 建構預測集
    # forecasting for 1 year from now.
    future = model.make_future_dataframe(periods=365)

    # 進行預測
    forecast = model.predict(future)

    figure = model.plot(forecast)
    st.pyplot(figure)


@staticmethod
def reset_plot():

    # Restore default parameters
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # Adjust a few parameters to liking
    matplotlib.rcParams['figure.figsize'] = (8, 5)
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['text.color'] = 'k'

def remove_weekends(dataframe):
    """
    Remove weekends from a dataframe
    """

    # Reset index to use ix
    dataframe = dataframe.reset_index(drop=True)

    weekends = []

    # Find all of the weekends
    for i, date in enumerate(dataframe['ds']):
        if (date.weekday()) == 5 | (date.weekday() == 6):
            weekends.append(i)

    # Drop the weekends
    dataframe = dataframe.drop(weekends, axis=0)

    return dataframe

 # Graph the effects of altering the changepoint prior scale (cps)


def changepoint_prior_analysis(stock, changepoint_priors=[0.01, 0.05, 0.1, 0.2], colors=['b', 'r', 'grey', 'gold']):

    # Training and plotting with specified years of data
    train = stock.copy(deep=True)

    # Iterate through all the changepoints and make models
    cols4 = st.columns(4)

    min_date = dtt.strptime(min(stock['ds']), '%Y-%m-%d')
    max_date = dtt.strptime(max(stock['ds']), '%Y-%m-%d')
    delta = max_date - min_date
    delta = delta.days

    # Prophet Setting
    weekly_seasonality = False
    daily_seasonality = False
    monthly_seasonality = True
    yearly_seasonality = True
    changepoints = None

    # get predictions
    for i, prior in enumerate(changepoint_priors):
        # Select the changepoint
        changepoint_prior_scale = prior

        # Create and train a model with the specified cps
        model = Prophet(daily_seasonality=daily_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        yearly_seasonality=yearly_seasonality,
                        changepoint_prior_scale=changepoint_prior_scale,
                        changepoints=changepoints)
        model.fit(train)

        future = model.make_future_dataframe(periods=365, freq='D')

        # Make a dataframe to hold predictions
        if i == 0:
            predictions = future.copy()

        future = model.predict(future)

        # Fill in prediction dataframe
        predictions['%.3f_yhat_upper' % prior] = future['yhat_upper']
        predictions['%.3f_yhat_lower' % prior] = future['yhat_lower']
        predictions['%.3f_yhat' % prior] = future['yhat']

    predictions['obs'] = train['y']
    st.header('prediction in 4 patterns')
    st.dataframe(predictions)

    # Remove the weekends
    # predictions = remove_weekends(predictions)

    # Plot set-up
    reset_plot()
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(1, 1)

    # Actual observations
    arrX = predictions['ds'].to_numpy()
    arrY = predictions['obs'].to_numpy()
    ax.plot(arrX, arrY, 'ko', ms=1, label='Observations')

    # Predictions
    color_dict = {prior: color for prior,
                  color in zip(changepoint_priors, colors)}

    for prior in changepoint_priors:

        # Plot the predictions themselves
        col_name = f"{prior:.3f}_yhat"
        arrY = predictions[col_name].to_numpy()
        arrY_upper = predictions[f'{col_name}_upper'].to_numpy()
        arrY_lower = predictions[f'{col_name}_lower'].to_numpy()

        # line
        ax.plot(arrX, arrY, linewidth=1.2,
                color=color_dict[prior], label=f'{col_name} prior scale', alpha=0.6)

        # Plot the uncertainty interval
        ax.fill_between(arrX, arrY_upper, arrY_lower, facecolor=color_dict[prior],
                        alpha=0.2, edgecolor='k', linewidth=0.3)

    # Plot labels
    plt.legend(loc=2, prop={'size': 10})
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.title('Effect of Changepoint Prior Scale')

    # plt.show()

    st.header("Changepoint_prior_analysis")

    st.pyplot(fig)
    fig.savefig(f"./03_png/test.png")


def add_sidebar():
    """note"""
    st.sidebar.write(f"TEST PAGE")


if __name__ == '__main__':

    try:
        print('-'*80)
        print('[INFO] START')

        st.title('Stock Analytic')
        # add_sidebar()

        # init setting
        com_no = '2330'
        startTime = '2014-07-01'
        endTime = '2018-08-01'

        # stock

        st.header("1. read data")
        cols1 = st.columns(2)
        # 1. from CSV
        stock = pd.read_csv(f'./01_obs/{filename}')
        # stock['Date'] = pd.to_datetime(stock['Date'])
        stock_ana = stock.copy(deep=True)
        stock_ana.set_index('Date', inplace=True)

        # 2. from DataReader
        # stock = pdr.DataReader(
        #         f'{com_no}.TW', 'yahoo', start=startTime, end='today')

        with cols1[0]:
            st.header("stock_ana")
            st.dataframe(stock_ana)

        # # st.dataframe(stock)
        stock['Date'] = pd.to_datetime(stock['Date'])
        new_df = pd.DataFrame(stock_ana['Adj Close']).reset_index().rename(
            columns={'Date': 'ds', 'Adj Close': 'y'})

        with cols1[1]:
            st.header("new_df")
            st.dataframe(new_df)

        # stock_analytic
        changepoint_prior_analysis(new_df, changepoint_priors=[
                                   0.001, 0.01, 0.1, 0.95])

    except Exception as err:
        print(f"[ERROR] {err}")
    finally:
        print('[INFO] END')
