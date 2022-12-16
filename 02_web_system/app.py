"""
===================================
  Stock Analysis and Prediction 
===================================
"""
import time
import streamlit as st
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# from matplotlib.ticker import FuncFormatter
# import seaborn as sns
# import pickle

# Stocker: A Stock Analysis and Prediction Toolkit using Additive Models
from modules.stocker import Stocker
# from stocker import Stocker

import pandas_datareader as pdr
from prophet import Prophet


# plot
# from streamlit_echarts import st_echarts
# import plotly.figure_factory as ff

st.set_page_config(layout="wide")

### Data Import ###
filename = '2330_20010101_20221216.csv'

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


def add_sidebar():
    """note"""
    st.sidebar.write(f"TEST PAGE")


if __name__ == '__main__':

    try:
        print('-'*30)
        print('[INFO] START')

        st.title('Stock Analytic')
        # add_sidebar()

        stock = get_obs_data()
        stock_ft = get_ft_data()

        # st.dataframe(stock)
        stock['Date'] = pd.to_datetime(stock['Date'])

        cols = st.columns(2)
        with cols[0]:
            st.header("obs")
            st.dataframe(stock)
        with cols[1]:
            st.header("ft")
            st.dataframe(stock_ft)

        # stock['AVG'] = (stock['Open'] + stock['Close'])/2
        # stock_ana = pd.concat([stock['Date'], stock['AVG'], stock_ft['yhat']], axis=1)
        # stock_ana.set_index('Date', inplace = True)

        #
        st.header("Analytic")
        # st.dataframe(stock_ana)
        # st.line_chart(stock_ana)
        
        com_no = '2330'
        startTime = '2000-01-01'
        endTime = '2022-12-16'
        df_stock = pdr.DataReader(
                f'{com_no}.TW', 'yahoo', start=startTime, end=endTime)
        new_df = pd.DataFrame(df_stock['Adj Close']).reset_index().rename(
                columns={'Date': 'ds', 'Adj Close': 'y'})
        stock_analytic(new_df)



    except Exception as err:
        print(f"[ERROR] {err}")
    finally:
        print('[INFO] END')
