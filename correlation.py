import datetime
from dotenv import load_dotenv
import json
import os
import requests
from typing import List, Optional
import json
import prices
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Loads 'env.shared' data into os.environ
load_dotenv(".env.shared")

from alert_types import SentimentSpikeAlert
from schemas import UnifiedTimeline, AggPeriod
from data import UptrendsWrapper

def main():
    ticker = 'TSLA'
    api = UptrendsWrapper(verbose=True)
    alerts = api.get_alert_objs(since_days_ago=7, until_days_ago=0)
    timelineRes = api.get_ticker_timeline_5m(ticker=ticker,
                                             since_dt=datetime.datetime.now() - datetime.timedelta(days=7),
                                             until_dt=datetime.datetime.now())
    print(len(timelineRes.data))

    tsla_alerts = api.get_alert_objs(
        since_days_ago=180,
        until_days_ago=0,
        ticker=ticker
    )
    tsla_prices = prices.get_ticker_daily_prices(
        ticker,
        start_dt=datetime.datetime.now() - datetime.timedelta(days=180),
        end_dt=datetime.datetime.now()
    )
    moving_avg_df = pd.DataFrame(columns=['date', 'rank', '9day_avg', '21day_avg', 'price'])

    ranks = []
    date = []

    for alert in tsla_alerts:
        ranks.append(json.loads(alert.model_dump_json(indent=2))['observed']['rank'])
        date.append(json.loads(alert.model_dump_json(indent=2))['alert_day'][0:10])
    ranks.reverse()
    date.reverse()
    moving_avg_df['rank'] = ranks
    moving_avg_df['date'] = date
    moving_avg_df['9day_avg'] = moving_avg_df['rank'].rolling(window=9, min_periods=1).mean()
    moving_avg_df['21day_avg'] = moving_avg_df['rank'].rolling(window=21, min_periods=1).mean()
    # 180 rows

    # direction_str
    # alert_day
    output = []
    for alert in tsla_alerts:
        #print(json.loads(alert.model_dump_json(indent=2, exclude={'timeline'})))
        day = json.loads(alert.model_dump_json(indent=2, exclude={'timeline'}))['alert_day']
        direction = json.loads(alert.model_dump_json(indent=2, exclude={'timeline'}))['direction_str']
        output.append([day[0:10], direction])

    for i in range(len(tsla_prices)):
        cur_day = str(tsla_prices[i].tick_datetime)[0:10]
        price = tsla_prices[i].close_price
        if cur_day in moving_avg_df['date'].values:
            row_index = moving_avg_df[moving_avg_df['date'] == cur_day].index[0]
            moving_avg_df.at[row_index, 'price'] = price

    output.reverse()
    date_list = []
    sentiment_signal = []
    price_signal = []
    indicator = "price"
    for p in output:
        date, sentiment = p
        price = moving_avg_df.loc[moving_avg_df["date"] == date][indicator].item()
        date_list.append(date)
        price_signal.append(price)
        sentiment_signal.append(1 if sentiment == "Bullish" else 0)

    print(price_signal)
    price_derivative = savgol_filter(price_signal, window_length=9, polyorder=5, deriv=1)

    for d in range(len(date_list)):
        sentiment = sentiment_signal[d]
        color = "green" if sentiment == 1 else "red"
        label = "bullish" if sentiment == 1 else "bearish"
        if not np.isnan(price_signal[d]):
            plt.scatter(date_list[d], price_signal[d], color=color, label=label)
        else:  # Linearly interpolate missing signal value
            approx_val = (price_signal[d-1]+price_signal[d+1])/2
            if not np.isnan(approx_val):
                plt.scatter(date_list[d], approx_val, color=color, label=label)
    plt.xticks([])
    # plt.yticks([])
    plt.xlabel("Date")
    plt.ylabel(indicator)
    legend_unique(plt)
    plt.show()
    #plt.savefig("direct-price.png")


def legend_unique(figure):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc='lower right')


if __name__ == "__main__":
    main()

