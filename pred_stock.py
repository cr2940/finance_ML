import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from datetime import date
from lxml import html
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
import argparse
from collections import OrderedDict

# to Fourier transform date feature
def sinuoid(number,typ):
    if typ == 'month':
        return np.sin(2*np.pi*number/12)
    elif typ == 'day':
        return np.sin(2*np.pi*number/31)
def cosine(number,typ):
    if typ == 'month':
        return np.cos(2*np.pi*number/12)
    elif typ == 'day':
        return np.cos(2*np.pi*number/31)

# to webscrape Yahoo Finance stock data
def get_headers():
    return {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9","accept-encoding": "gzip, deflate, br","accept-language": "en-GB,en;q=0.9,en-US;q=0.8,ml;q=0.7","cache-control": "max-age=0","dnt": "1","sec-fetch-dest": "document","sec-fetch-mode": "navigate","sec-fetch-site": "none","sec-fetch-user": "?1","upgrade-insecure-requests": "1","user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36"}
def parse(ticker):
    url = "http://finance.yahoo.com/quote/%s?p=%s" % (ticker, ticker)
    response = requests.get(url, verify=True, headers=get_headers(), timeout=30)
    # print("Parsing %s" % (url))
    parser = html.fromstring(response.text)
    summary_table = parser.xpath('//div[contains(@data-test,"summary-table")]//tr')
    summary_data = OrderedDict()
    other_details_json_link = "http://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(ticker)
    summary_json_response = requests.get(other_details_json_link, headers=get_headers())
    try:
        json_loaded_summary = json.loads(summary_json_response.text)
        summary = json_loaded_summary["quoteSummary"]["result"][0]
        y_Target_Est = summary["financialData"]["targetMeanPrice"]['raw']
        earnings_list = summary["calendarEvents"]['earnings']
        eps = summary["defaultKeyStatistics"]["trailingEps"]['raw']
        datelist = []
        for i in earnings_list['earningsDate']:
            datelist.append(i['fmt'])
            earnings_date = ' to '.join(datelist)
        if len(datelist)==0:
            earnings_date = "None"
        for table_data in summary_table:
            raw_table_key = table_data.xpath('.//td[1]//text()')
            raw_table_value = table_data.xpath('.//td[2]//text()')
            table_key = ''.join(raw_table_key).strip()
            table_value = ''.join(raw_table_value).strip()
            summary_data.update({table_key: table_value})
            summary_data.update({'1y Target Est': y_Target_Est, 'EPS (TTM)': eps,'Earnings Date': earnings_date, 'ticker': ticker,'url': url})
        return summary_data
    except ValueError:
        print("Failed to parse json response")
        return {"error": "Failed to parse json response"}
    except:
        return {"error": "Unhandled Error"}

# User defined Stock/ETF tick mark, and searching the data for it
prod = input("Enter the ticker of either the Stock or ETF: ")
print(' ............................ ')
fname_stock = './archive/Stocks/'+ prod +'.us.txt'
fname_etf = './archive/ETFs/' + prod + '.us.txt'
if os.path.isfile(fname_stock):
    data_prod = pd.read_csv(fname_stock)
elif os.path.isfile(fname_etf):
    data_prod = pd.read_csv(fname_etf)
else:
    raise TypeError('No such ETF or Stock found')

# Using Pandas to make data frame with appropriate data features
data_prod = data_prod.dropna()
data_prod['Date'] = pd.to_datetime(data_prod['Date'])
data_prod['yrs'] = data_prod['Date'].dt.year
data_prod['mon_sin'] = np.sin(2*np.pi*data_prod['Date'].dt.month/12)
data_prod['mon_cos'] = np.cos(2*np.pi*data_prod['Date'].dt.month/12)
data_prod['day_sin'] = np.sin(2*np.pi*data_prod['Date'].dt.day/31)
data_prod['day_cos'] = np.cos(2*np.pi*data_prod['Date'].dt.day/31)
## if you will change attributes here, change today_data as well
def attribute_maker(yr,mon,day,opn,vol):
    return [yr,sinuoid(mon,"month"),cosine(mon,"month"),sinuoid(day,"day"),cosine(day,"day"),opn,vol]
X = data_prod[['yrs','mon_sin','mon_cos','day_sin','day_cos','Open','Volume']]
Y = data_prod['Close']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# fit Lasso Linear Regression model to data
lasso_LR = linear_model.Lasso(alpha=1)
lasso_LR.fit(X_train,y_train)
y_pred = lasso_LR.predict(X_test)
print(' ............................ ')
print("The Lasso linear regression model has accuracy of :" ,lasso_LR.score(X_test,y_test), " for stock/ETF ",prod)
print("Now retrieving today's open price data and volume....")

# Web scrape stock data to get today's data features before close
summary_data = parse(prod)
vol = int(summary_data['Volume'].replace(',','')); opn = float(summary_data['Open'].replace(',','')); max_day = float(summary_data['Day\'s Range'].split(sep=' - ')[1].replace(',','')); min_day = float(summary_data['Day\'s Range'].split(sep=' - ')[0].replace(',',''))
today = date.today()
today_data = [today.year, today.month, today.day, opn,vol]
todays_atts = np.reshape(attribute_maker(*today_data),(1,len(attribute_maker(*today_data))))

# predict using model:
today_pred = lasso_LR.predict(todays_atts)
print(' ............................ ')
print("Today's open price was ",opn," and ")
print("today's predicted closing price is: ", today_pred)
print("so we suggest you",'sell' if today_pred>opn else 'keep', "the stock/ETF",prod, 'for short term capital gain!' if today_pred>opn  else 'and wait a little longer')
