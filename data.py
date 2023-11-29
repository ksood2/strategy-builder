
import datetime
from dotenv import load_dotenv
import json
import os
import requests
from typing import List, Optional

# Loads 'env.shared' data into os.environ
load_dotenv(".env.shared")

from alert_types import SentimentSpikeAlert
from schemas import UnifiedTimeline, AggPeriod


class UptrendsWrapper:
    def __init__(self, verbose:bool = False):
        self.verbose = verbose
        self.api_key = os.environ.get("BASIC_UPTRENDS_API_KEY")
        self.dev_api_base = os.environ.get("DEV_UPTRENDS_BASE_API")
        self.prod_api_base = os.environ.get("PROD_UPTRENDS_BASE_API")

        self.api_base = self.prod_api_base
        
        # Let's grab a session ID
        sess_resp = requests.get(self.api_base+f"/sessions/?user_key={self.api_key}")
        if sess_resp.status_code != 200:
            #print(f"[ERROR][UW.__init__] Got status code {sess_resp.status_code} from API ({self.api_base})")
            #print(sess_resp.text)
            self.session_id = "sess-placeholder"
        else:
            sess_d = sess_resp.json()
            self.session_id = sess_d["session_id"]
            if self.verbose:
                print(f"[INFO][UW.__init__] Got session ID {self.session_id}")
        
    def get_alert_objs(self, since_days_ago: int=7, until_days_ago: int=0, max_results:int=200, ticker: Optional[str]=None):
        arg_str = f"since={since_days_ago}&until={until_days_ago}&max_results={max_results}"
        if ticker is not None:
            ticker = ticker.upper()
            # add in the ticker arg
            arg_str += f"&ticker={ticker}"

        alert_resp = requests.get(f"{self.api_base}/events/feed/alerts?{arg_str}")
        alert_d = alert_resp.json()
        alerts_lst: List[dict] = alert_d["data"]

        alerts_objs = [SentimentSpikeAlert.model_validate(x) for x in alerts_lst]
        
        if self.verbose:
            print(f"[INFO][UW.get_alert_objs] Got {len(alerts_objs)} alerts, ticker set to '{ticker}'")
        return alerts_objs
    
    def _get_ticker_timeline_generic(
            self,
            ticker: str,
            since_dt: datetime.datetime,
            until_dt: datetime.datetime,
            period: AggPeriod
    ) -> UnifiedTimeline:
        """> Returns a UnifiedTimeline object for a given ticker, between these datetimes. 
        NOTE: Each datapoint in UnifiedTimeline.data is an aggregation chosen by the 'period' arg
            containing the sentiment data, price data, and events data that happened in each window
        """
        ticker = ticker.upper()
        since_dt = since_dt.astimezone(datetime.timezone.utc)
        until_dt = until_dt.astimezone(datetime.timezone.utc)

        since_dt_str = since_dt.isoformat().replace("+00:00", "Z")
        until_dt_str = until_dt.isoformat().replace("+00:00", "Z")

        timeline_resp = requests.get(
            f"{self.api_base}/stocks/{ticker}/timeline?since={since_dt_str}&until={until_dt_str}&period={period.value}"
        )
        if timeline_resp.status_code != 200:
            print(f"[ERROR][UW.get_ticker_timeline_5m] Got status code {timeline_resp.status_code} from API ({self.api_base})")
            print(timeline_resp.text)
            return None
        
        raw_data = timeline_resp.json()
        timeline_obj = UnifiedTimeline.model_validate(raw_data)
        return timeline_obj
        
    def get_ticker_timeline_5m(
            self, 
            ticker: str, 
            since_dt: datetime.datetime, 
            until_dt: datetime.datetime
        ) -> UnifiedTimeline:
        """> Returns a UnifiedTimeline object for a given ticker, between these datetimes. 
        NOTE: Each datapoint in UnifiedTimeline.data is a 5-minute aggregation 
            containing the sentiment data, price data, and events data that happened in that 5-minute window
        """
        timeline_obj = self._get_ticker_timeline_generic(ticker, since_dt, until_dt, AggPeriod.five_minute)
        return timeline_obj
    
    def get_ticker_timeline_hourly(
            self, 
            ticker: str, 
            since_dt: datetime.datetime, 
            until_dt: datetime.datetime
        ) -> UnifiedTimeline:
        """> Returns a UnifiedTimeline object for a given ticker, between these datetimes. 
        NOTE: Each datapoint in UnifiedTimeline.data is an hourly aggregation 
            containing the sentiment data, price data, and events data that happened in that 1-hour window
        """
        timeline_obj = self._get_ticker_timeline_generic(ticker, since_dt, until_dt, AggPeriod.hourly)
        return timeline_obj
    
    def get_ticker_timeline_daily(
            self, 
            ticker: str, 
            since_dt: datetime.datetime, 
            until_dt: datetime.datetime
        ) -> UnifiedTimeline:
        """> Returns a UnifiedTimeline object for a given ticker, between these datetimes. 
        NOTE: Each datapoint in UnifiedTimeline.data is a daily aggregation 
            containing the sentiment data, price data, and events data that happened in that day's window
        """
        timeline_obj = self._get_ticker_timeline_generic(ticker, since_dt, until_dt, AggPeriod.daily)
        return timeline_obj


if __name__=="__main__":
    api = UptrendsWrapper(verbose=True)
    
    for agg_func in [api.get_ticker_timeline_5m, api.get_ticker_timeline_hourly, api.get_ticker_timeline_daily]:
        timeline_obj = agg_func(
            ticker="TSLA", 
            since_dt=datetime.datetime.now()-datetime.timedelta(days=7), 
            until_dt=datetime.datetime.now()-datetime.timedelta(days=6, hours=10)
        )
        print(f"Got {len(timeline_obj.data)} ticks for {timeline_obj.ticker}, aggregated at {timeline_obj.agg_period.value}")
        print(f"  > min date: {timeline_obj.data[0].tick_datetime.isoformat()}")
        print(f"  > max date: {timeline_obj.data[-1].tick_datetime.isoformat()}")

    alerts = api.get_alert_objs(since_days_ago=7, until_days_ago=0)
    tsla_alerts = api.get_alert_objs(
        since_days_ago=60,
        until_days_ago=0,
        ticker="TSLA"
    )
    # \/ Uncomment this to see the example alert JSON
    #print(tsla_alerts[0].model_dump_json(indent=2, exclude={"timeline"}))


