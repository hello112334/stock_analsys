"""
===================================
  Stock Analysis and Prediction
===================================
"""
# Modules
from bs4 import BeautifulSoup
import requests
from pytrends.request import TrendReq
import pandas_datareader as pdr
from prophet import Prophet
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime as dtt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
import time
import warnings  # remove warnings
warnings.filterwarnings('ignore')

pytrends = TrendReq(hl='en-US', tz=360)


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


@staticmethod
def reset_plot():
    """note"""

    # Restore default parameters
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # Adjust a few parameters to liking
    matplotlib.rcParams['figure.figsize'] = (8, 5)
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['text.color'] = 'k'


def changepoint_prior_analysis(stock, changepoint_priors=[0.01, 0.05, 0.1, 0.2], colors=['b', 'r', 'grey', 'gold'], ft_days=365):
    """
    Graph the effects of altering the changepoint prior scale (cps)
    """
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

        future = model.make_future_dataframe(periods=ft_days, freq='D')

        # Make a dataframe to hold predictions
        if i == 0:
            predictions = future.copy()

        future = model.predict(future)

        # Fill in prediction dataframe
        predictions['%.3f_yhat_upper' % prior] = future['yhat_upper']
        predictions['%.3f_yhat_lower' % prior] = future['yhat_lower']
        predictions['%.3f_yhat' % prior] = future['yhat']

    predictions['obs'] = train['y']

    # Check the data
    # st.header('prediction in 4 patterns')
    # st.dataframe(predictions)

    # Plot set-up
    # reset_plot()
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()

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

    #
    ax.set_ylim(-50, 1200)

    # Plot labels
    plt.legend(loc=2, prop={'size': 10})
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.title('Effect of Changepoint Prior Scale')

    # plt.show()

    # st.header("Changepoint_prior_analysis")
    # fig.savefig(f"./03_png/test.png")
    # st.pyplot(fig)
    return fig, predictions


def add_sidebar():
    """note"""
    st.sidebar.write(f"TEST PAGE")


def get_google_trends(start, end):
    """note"""
    # scrawing
    # url = f'https://trends.google.com/trends/explore?date={start}%20{end}'

    # headers = requests.utils.default_headers()
    # headers.update({
    #     'accept': 'application/json',
    #     'cache-control': 'no-cache',
    #     'sec-ch-ua-platform': "Windows",
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    # })
    # html_text = requests.get(url, headers=headers).text
    # soup = BeautifulSoup(html_text, 'html.parser')
    # st.write(html_text)

    # data = soup.findAll(
    #     'div', {'class': 'progress-label-wrapper'})

    # pytrends

    st.write(data)


def check_error(obs_ft_data, max_val, title):
    """
    1. plot
    2. check correlation coefficient and slope
    3. RMSE: Root Mean Squared Error
    """
    # st.dataframe(obs_ft_data)

    # reset_plot()
    # plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()

    arr_obs = obs_ft_data['obs'].to_numpy()
    arr_ft = obs_ft_data['ft'].to_numpy()

    
    # RMSE
    rmse = np.sqrt(mean_squared_error(arr_obs, arr_ft))
    # print(f'RMSE : {rmse:.3f}')

    # corr
    corr = np.corrcoef(arr_obs, arr_ft)

    # scatter obs ft
    ax.scatter(arr_obs, arr_ft, c='#1f77b4', alpha=0.9, label=f'RMSE : {rmse:.3f}')

    # slope
    model_lr = LinearRegression(fit_intercept=False)
    model_lr.fit(arr_obs.reshape(-1, 1), arr_ft.reshape(-1, 1))
    slope = model_lr.coef_
    ax.plot([0, max_val], [0, max_val], linewidth=1.0,
            color='#FF1493', label=f'corr: {corr[0,1]}', alpha=1.0)
    ax.plot([0, max_val], [0, max_val*slope], linewidth=1.0,
            color='#888888', label=f'slope: {slope[0][0]}', alpha=0.8)

    #
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # plt.text(0.1*max_val, 0.3*max_val, f'corr: {corr}')

    # Plot labels
    plt.legend(loc=2, prop={'size': 10})
    plt.xlabel('obs')
    plt.ylabel('ft')
    plt.title(f'{title}')
    fig.tight_layout()

    st.pyplot(fig)
    plt.close()


def Spearman_analysis():
    """note"""


def show_obs_all(stock):
    """note"""
    # 1. plot
    # fig, ax = plt.subplots()
    # arrX = stock['Date'].to_numpy()
    # arrY = stock['Adj Close'].to_numpy()

    # # scatter obs ft
    # ax.plot(arrX, arrY, 'ko', ms=1, label='Observations')

    # # Plot labels
    # plt.legend(loc=1, prop={'size': 10})
    # plt.xlabel('obs')
    # plt.ylabel('price')
    
    # plt.grid()
    # myFmt = DateFormatter("%y")
    # ax.xaxis.set_major_formatter(myFmt)
    # fig.autofmt_xdate()
    # fig.tight_layout()

    # fig.savefig(
    #     f"./03_png/all_period.png")  # Plot
    # plt.close()
    
    # 2. Check data with event
    # st.header(f'{startTime} - {endTime}')
    chart_data = stock[(stock['Date'] > '2000-07-01')
                        & (stock['Date'] < '2022-12-16')]
    chart_data['Date'] = pd.to_datetime(chart_data['Date'])
    chart_data = chart_data.set_index('Date')
    st.line_chart(chart_data['Adj Close'])


if __name__ == '__main__':

    try:
        print('-'*80)
        print('[INFO] START')

        st.set_page_config(layout="wide")
        st.title('Stock Analytic')
        # add_sidebar()

        # init setting
        com_no = '2330' # Stock number
        # reset_plot()

        # stock
        st.header("1. Get Data")

        # 1. from CSV
        stock = get_obs_data()

        # 2. read from DataReader
        # stock = pdr.DataReader(
        #         f'{com_no}.TW', 'yahoo', start=startTime, end='today')

        # 3. scrawing form yahoo
        # reference to script

        # All data
        st.header(f"Stock Data: {com_no}")

        with st.expander("See data"):
            st.dataframe(stock)
        show_obs_all(stock) # plot all period data
        # raise Exception("test")

        # analysis period set
        # dataset = []
        dataset = [{'start': '2014-07-01', 'end': '2018-07-01'},
                   {'start': '2018-07-01', 'end': '2021-12-10'},
                   {'start': '2020-07-01', 'end': '2021-12-10'}]

        if dataset:
            cols = st.columns(len(dataset))
            for i in range(len(dataset)):
                startTime = dataset[i]['start']
                endTime = dataset[i]['end']

                # filter data between start and end
                stock_ana = stock[(stock['Date'] > startTime)
                                  & (stock['Date'] < endTime)]
                stock_ana.set_index('Date', inplace=True)

                # convert col names
                new_df = pd.DataFrame(stock_ana['Adj Close']).reset_index().rename(
                    columns={'Date': 'ds', 'Adj Close': 'y'})

                # stock_analytic
                changepoint_prior = [0.001, 0.01, 0.1, 1.000]
                ft_day = 365  # 1 year
                fig, predictions = changepoint_prior_analysis(
                    new_df, changepoint_priors=changepoint_prior, ft_days=ft_day)

                # check error
                with cols[i]:
                    st.write(f'#### {startTime} - {endTime}, ft: {ft_day} days')
                    st.pyplot(fig)
                    plt.close()

                    for prior in changepoint_prior:
                        # st.dataframe(predictions)
                        tmp_date = dtt.strptime(endTime, "%Y-%m-%d")
                        endTime_ft = tmp_date + datetime.timedelta(days=ft_day)
                        endTime_ft = endTime_ft.strftime("%Y-%m-%d")
                        # print(f'LAST DAY: {endTime_ft}')

                        # OBS
                        obs_data = stock[['Date', 'Adj Close']]
                        obs_data = obs_data[(obs_data['Date'] > endTime) & (
                            obs_data['Date'] < endTime_ft)]
                        obs_data = obs_data.rename(
                            columns={'Adj Close': 'obs', 'Date': 'ds'})
                        obs_data['ds'] = pd.to_datetime(obs_data['ds'])
                        obs_data.set_index('ds', inplace=True)
                        obs_data = obs_data.reset_index(drop=True)

                        # FT
                        ft_data = predictions[['ds', f'{prior:.3f}_yhat']]
                        ft_data = ft_data[(ft_data['ds'] > endTime) & (
                            ft_data['ds'] < endTime_ft)]
                        ft_data = ft_data.rename(
                            columns={f'{prior:.3f}_yhat': 'ft'})
                        ft_data.set_index('ds', inplace=True)
                        ft_data = ft_data.reset_index(drop=True)

                        # st.dataframe(obs_data)
                        # st.dataframe(ft_data)

                        all_data = pd.concat([obs_data, ft_data], axis=1)
                        all_data = all_data[(all_data['obs'].notnull()) & (all_data['ft'].notnull()) ]
                        max_val_obs = max(all_data['obs'])
                        max_val_ft = max(all_data['ft'])
                        max_val = max(max_val_obs, max_val_ft)

                        max_val = int(max_val // 100) * 100 + 200
                        check_error(all_data, max_val, f'{prior:.3f}_yhat')

        # check keyword from google trends
        st.header('9.google trends')
        # https://trends.google.com/trends/

        dataset = pd.DataFrame()
        # keywords 5 words/1 time
        list1 = ["machine", "Football", "New Year"]
        list2 = ["TSMC", "Taiwan Semiconductor Manufacturing",
                 "Semiconductor", "Manufacturing"]
        list3 = ["Weather", "Earthquake", "Natural Disaster",
                 "Central Meteorological Bureau of the Ministry of Communications"]
        list4 = ["stock", "stock market",
                 "Mark Six", "invoice"]
        list5 = ["Quarantine", "oil prices", "vaccines", "EVA Air strike"]
        list_all = [list1, list2, list3, list4, list5]

        pytrends = TrendReq(hl='US')
        for item in list_all:
            pytrends.build_payload(item, timeframe='all', geo='TW')
            df = pytrends.interest_over_time()
            if not df.empty:
                data = df.drop(labels=['isPartial'], axis='columns')
                # dataset.append(data)
                dataset = pd.concat([dataset, data], axis=1)

        # result = pd.DataFrame(dataset)
        st.dataframe(dataset)

        ###
        st.dataframe(new_df)
        stock_obs = new_df.rename(columns={'ds': 'Date', 'y': 'Price'})

        # coin_data = coin_data.drop(['Volume'], axis=1)
        # coin_data.columns = ['Date', 'Price']
        # trendslen = 0
        # with open('price-daily.csv', 'r') as f:
        #     trendslen = len(f.readlines())
        # for i in range(0, trendslen - 1):
        #     # shorten in 500 time
        #     coin_data.loc[i, 'Price'] = coin_data.loc[i, 'Price'] / 500
        #     coin_data.loc[i, 'Price'] = format(
        #         coin_data.loc[i, 'Price'], '.3f')
        # last = pd.concat([coin_data, trends_data], axis=1)
        # last.to_csv('price-and-trends.csv', index=False)

    except Exception as err:
        print(f"[ERROR] {err}")
    finally:
        print('[INFO] END')
        plt.close('all')
