
import datetime
from dotenv import load_dotenv
from typing import Iterator, Optional, Union, List
import os
import pytz
import requests

from schemas import TickerPricesResponse, PriceAgg
load_dotenv(".env.shared")


def get_ticker_daily_prices(ticker: str, start_dt: datetime.datetime, end_dt: datetime.datetime):
    arg_str=f"?since={start_dt.isoformat()}&until={end_dt.isoformat()}"
    api_resp = requests.get(os.environ.get("PROD_UPTRENDS_BASE_API", "https://api.uptrends.ai")+f"/stocks/{ticker.upper()}/prices{arg_str}")
    prices_resp = api_resp.json()
    prices_obj = TickerPricesResponse.model_validate(prices_resp)
    return prices_obj.data


def get_ticker_price_change_percent(ticker: str, start_dt: datetime.datetime, end_dt: datetime.datetime):
    #price_objs = get_ticker_daily_prices(tick=ticker, start_day=start_dt, end_day=end_dt)
    price_objs: List[PriceAgg] = get_ticker_daily_prices(ticker, start_dt, end_dt)

    if price_objs is None or len(price_objs) < 2:
        return 0.0
    
    prices_asc = sorted(price_objs, key=lambda x: x.tick_datetime)
    open_price = prices_asc[0].open_price
    close_price = prices_asc[-1].close_price
    return (close_price - open_price) / open_price


if __name__=="__main__":
    tsla_5d = get_ticker_price_change_percent(
        "TSLA", 
        start_dt=datetime.datetime.now()-datetime.timedelta(days=5),
        end_dt=datetime.datetime.now()
    )

    tsla_30d = get_ticker_price_change_percent(
        "TSLA", 
        start_dt=datetime.datetime.now()-datetime.timedelta(days=30),
        end_dt=datetime.datetime.now()
    )

    tsla_180d = get_ticker_price_change_percent(
        "TSLA", 
        start_dt=datetime.datetime.now()-datetime.timedelta(days=180),
        end_dt=datetime.datetime.now()
    )
    print("TSLA:")
    print(f"    5d: {'+' if tsla_5d > 0 else ''}{round(tsla_5d*100, 1)}%")
    print(f"   30d: {'+' if tsla_30d > 0 else ''}{round(tsla_30d*100, 1)}%")
    print(f"  180d: {'+' if tsla_180d > 0 else ''}{round(tsla_180d*100, 1)}%")
