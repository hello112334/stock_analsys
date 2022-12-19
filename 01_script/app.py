"""
================================
  Stock Analytic
================================
"""
# basic
import datetime as datetime
from datetime import datetime as dtt
from datetime import date
import time
from dateutil.relativedelta import relativedelta

# modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# stock data
import twstock
import pandas_datareader as pdr

# scrawing
import requests
from bs4 import BeautifulSoup

# analysis/analytic
from prophet import Prophet
from sklearn import metrics

# 上市編號
com_no = "2330"
# datetime_from = [2000, 1]
# get_twstock = twstock.Stock(com_no)

date_today = 0


def get_history_twstock1():
    """note"""
    # 近期
    # price_6207 = get_stock.price[-5:]       # 近五日之收盤價
    # high_6207 = get_stock.high[-5:]         # 近五日之盤中高點
    # low_6207 = get_stock.low[-5:]           # 近五日之盤中低點
    # date_6207 = get_stock.date[-5:]         # 近五日的日期

    # 獲取 yyyy 年 mm 月至今日之股票資料
    # get_stock = get_twstock.fetch_from(datetime_from[0], datetime_from[1])
    # get_stock_pd = pd.DataFrame(get_stock)
    # get_stock_pd = get_stock_pd.set_index('date')

    # fig = plt.figure(figsize=(10, 6))
    # plt.plot(get_stock_pd.close, '-' , label="收盤價")
    # plt.plot(get_stock_pd.open, '-' , label="開盤價")
    # plt.title(f'{datetime_from[0]} - {dtt.today().year}',loc='right')

    # loc->title的位置
    # plt.xlabel('日期')
    # plt.ylabel('收盤價')
    # plt.grid(True, axis='y')
    # plt.legend()
    # fig.savefig(f"./02_png/{com_no}_19800101_{date_today}_open_close.png")


def get_history_yahoo1():
    """
    歷史資料抓取 YAHOO
    --------
    參數名稱   描述
    Open      開盤價
    High      最高價
    Low       最低價
    Close     收盤價
    Volume    交易量
    Adj Close 經過調整的收盤價
    """
    startTime = '2018-10-01'
    endTime = '2018-10-30'
    df_2330 = pdr.DataReader('2330.TW', 'yahoo', startTime, endTime)


def get_history_yahoo2():
    """
    歷史資料抓取 twstock
    --------
    LIMIT: 3 times/5 seconds
    """
    print("[INFO] get_history")

    # ----- 時間
    # 開始
    startTime = '2014-07-01'
    startTime_str = startTime.replace("-", "")

    # 終了
    endTime = '2018-08-01'
    endTime_str = endTime.replace("-", "")
    # endTime = getToday(hyphen='yes')
    # endTime_str = getToday(hyphen='no')

    # ----- Get Data
    df_stock = pdr.DataReader(
        f'{com_no}.TW', 'yahoo', start=startTime, end=endTime)
    df_stock.to_csv(
        f"./01_obs/{com_no}_{startTime_str}_{endTime_str}.csv")  # CSV

    new_df = pd.DataFrame(df_stock['Adj Close']).reset_index().rename(
        columns={'Date': 'ds', 'Adj Close': 'y'})

    # csv
    stock_res = (get_stock_pd.close + get_stock_pd.open) / 2
    stock_res.to_csv(f"./01_data/{com_no}_{startTime_str}_{endTime_str}.csv")

    print("[INFO] analysis")
    # 定義模型 Facebook
    model = Prophet()

    # 訓練模型
    model.fit(new_df)

    # 建構預測集
    # forecasting for 1 year from now.
    future = model.make_future_dataframe(periods=365)

    # 進行預測
    forecast = model.predict(future)

    figure = model.plot(forecast)
    forecast.to_csv(
        f"./02_ft/{com_no}_{startTime_str}_{endTime_str}.csv")  # CSV
    figure.savefig(
        f"./03_png/{com_no}_{startTime_str}_{endTime_str}.png")  # Plot


def get_history_yahoo3(name, data_source, start, end):
    """note"""
    print(f'GET YAHOO STOCK: {start}-{end}')

    set_start = dtt.strptime(start, '%Y-%m-%d')
    tmp_end = dtt.strptime(end, '%Y-%m-%d')
    set_end = set_start + relativedelta(months=3)

    headers = requests.utils.default_headers()
    headers.update({
        'accept': 'application/json',
        'cache-control': 'no-cache',
        'sec-ch-ua-platform': "Windows",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    })

    stock = [[], [], [], [], [], [], []]
    stop_flg = False
    while set_end <= tmp_end:

        start_time = int(time.mktime(set_start.timetuple()))  # 1404154800
        end_time = int(time.mktime(set_end.timetuple()))  # 1533149999

        gm_url = f'https://finance.yahoo.com/quote/{name}/history?period1={start_time}&period2={end_time}&interval=1d&frequency=1d&filter=history'
        html_text = requests.get(gm_url, headers=headers).text
        soup = BeautifulSoup(html_text, 'html.parser')

        col = 0
        count = 0
        getDate = soup.findAll(
            'td', {'class': 'Py(10px) Ta(start) Pend(10px)'})
        for div in soup.findAll('td', {'class': 'Py(10px) Pstart(10px)'}):
            # print(div.text.strip())
            tmp_val = div.text.strip()
            if col == 0:
                tmp_date = dtt.strptime(
                    getDate[count].text.strip(), '%b %d, %Y')  # Sep 30, 2014
                stock[6].append(tmp_date)
                count += 1
            if col == 5:
                tmp_val = div.text.strip().replace(',', '')

            stock[col].append(tmp_val)
            col += 1
            col = 0 if col == 6 else col

        set_start = set_start + relativedelta(months=3)
        set_end = set_end + relativedelta(months=3)
        if stop_flg:
            break
        if set_end > tmp_end:
            set_end = tmp_end
            stop_flg = True

    yahoo_df = pd.DataFrame(
        {'Date': stock[6], 'Open': stock[0], 'High': stock[1], 'Low': stock[2], 'Close': stock[3], 'Adj Close': stock[4], 'Volumn': stock[5]})
    yahoo_df = yahoo_df.replace("-", 0)
    yahoo_df = yahoo_df.sort_values(by='Date')


    res_df = yahoo_df[(yahoo_df['Open'] != 0) & (yahoo_df['Adj Close'] != 0)]

    return res_df


def get_realtime():
    """
    抓取及時資料
    --------
    """
    get_stock_real = twstock.realtime.get('6207')

    # 抓取多個股票的方式 twstock.realtime.get(['2330', '2337', '2409'])
    get_stock_real


def getToday(hyphen='no'):
    """
    獲得今日日期
    ---------
    parameter
    1. hyphen: 是否追加 -

    return string: yyyymmdd or yyyy-mm-dd
    """

    tmp_today = dtt.today()
    if hyphen == 'no':
        return f"{tmp_today.year}{tmp_today.month}{tmp_today.day}"
    else:
        return f"{tmp_today.year}-{tmp_today.month}-{tmp_today.day}"


if __name__ == "__main__":

    try:
        date_today = getToday()
        # get_history_yahoo2()
        # get_realtime()

        # -------------------------
        # 開始
        startTime = '2000-01-01'
        startTime_str = startTime.replace("-", "")

        # 終了
        endTime = '2022-12-18'
        endTime_str = endTime.replace("-", "")

        yahoo_df = get_history_yahoo3(
            f'{com_no}.TW', 'yahoo', start=startTime, end=endTime)
        
        yahoo_df.to_csv(f"./01_obs/{com_no}_{startTime_str}_{endTime_str}.csv")

    except Exception as err:
        print(f"[INFO] {err}")
