import twstock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# get data
import pandas_datareader as pdr

# from matplotlib.font_manager import fontManager
# fontManager.addfont('./TaipeiSansTCBeta-Regular.ttf')
# plt.rc('font', family='Sans TC Beta')

import datetime as datetime
from datetime import datetime as dtt


# analysis
from prophet import Prophet
from sklearn import metrics

# pandas_datareader YAHOO
# import pandas_datareader as pdr
# df_2330 = pdr.DataReader('2330.TW', 'yahoo')

# 這是抓取歷史資料
com_no = "2330"
# datetime_from = [2000, 1]
# get_twstock = twstock.Stock(com_no)

date_today = 0

def get_history_yahoo():
    """
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

def get_history():
    """
    twstock

    LIMIT: 3 times/5 seconds
    """
    print("[INFO] get_history")

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

    # 
    startTime = '2001-01-01'
    endTime = '2022-12-13'
    df_stock = pdr.DataReader(f'{com_no}.TW', 'yahoo', start=startTime, end=endTime)
    new_df_2492 = pd.DataFrame(df_stock['Adj Close']).reset_index().rename(columns={'Date':'ds', 'Adj Close':'y'})

    # csv
    # stock_res = (get_stock_pd.close + get_stock_pd.open) / 2
    # stock_res.to_csv(f"./01_data/{com_no}_19800101_{date_today}.csv")

    print("[INFO] analysis")
    # 
    new_df_2492['y'] = np.log(new_df_2492['y'])

    # 定義模型
    model = Prophet()

    # 訓練模型
    model.fit(new_df_2492)

    # 建構預測集
    future = model.make_future_dataframe(periods=365) #forecasting for 1 year from now.

    # 進行預測
    forecast = model.predict(future)

    figure=model.plot(forecast)
    figure.savefig(f"./02_png/{com_no}.png")

def get_realtime():
    get_stock_real = twstock.realtime.get('6207')

    # 抓取多個股票的方式 twstock.realtime.get(['2330', '2337', '2409'])
    get_stock_real

def getToday():

    tmp_today = dtt.today()
    tmp_res = f"{tmp_today.year}{tmp_today.month}{tmp_today.day}"
    return tmp_res


if __name__ == "__main__":

    try:
        date_today = getToday()
        get_history()
        
        # get_realtime()

    except Exception as err:
        print(f"[INFO] {err}")
