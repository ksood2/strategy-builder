
import datetime
from polygon import RESTClient
from polygon.rest.models import TickerSnapshot as PolygonTickerSnapshot, Agg
from typing import Iterator, Optional, Union, List
import pytz

from schemas import Baseline

polygon_client = RESTClient(api_key="0Gc1iUzgI9N5IrAT4fXpq8pLQpvrLEJr")

class PriceDay(Baseline):
    avg_price: float
    tot_volume: int
    tick_dt: datetime.datetime
    num_price_ticks: Optional[int] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None


def get_ticker_daily_prices(
        tick: str, 
        start_day: datetime.datetime=datetime.datetime.today(), 
        end_day: datetime.datetime=datetime.datetime.today()
    ) -> Union[List[PriceDay], None]:
    start_date_str = start_day.strftime("%Y-%m-%d")
    end_date_str = end_day.strftime("%Y-%m-%d")

    res = polygon_client.list_aggs(
        ticker=tick,
        multiplier=1,
        timespan="day",
        from_=start_date_str,
        to=end_date_str
    )
    if isinstance(res, Iterator):
        try:
            all_aggs = [x for x in res]
        except KeyError:
            print(f"[polygon_error] {tick}: {start_day.isoformat()} <> {end_day.isoformat()}")
            return None

        found_prices = []
        for agg in all_aggs:
            timestamp_time = datetime.datetime.fromtimestamp(agg.timestamp / (10**3), tz=pytz.utc)

            price_obj = PriceDay(
                avg_price=agg.vwap,
                tot_volume=int(agg.volume),
                tick_dt=timestamp_time,
                num_price_ticks=1,
                open_price=agg.open,
                close_price=agg.close
            )
            found_prices.append(price_obj)
        return found_prices
    
    print(type(res))
    print(res)
    return None


def get_ticker_price_change_percent(ticker: str, start_dt: datetime.datetime, end_dt: datetime.datetime):
    price_objs = get_ticker_daily_prices(tick=ticker, start_day=start_dt, end_day=end_dt)
    if price_objs is None or len(price_objs) < 2:
        return 0.0
    
    prices_asc = sorted(price_objs, key=lambda x: x.tick_dt)
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
