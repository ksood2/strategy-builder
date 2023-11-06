
from dotenv import load_dotenv
import os
import requests
from typing import List, Optional

# Loads 'env.shared' data into os.environ
load_dotenv(".env.shared")

from alert_types import SentimentSpikeAlert


class UptrendsWrapper:
    def __init__(self, verbose:bool = False):
        self.verbose = verbose
        self.api_key = os.environ.get("BASIC_UPTRENDS_API_KEY")
        self.dev_api_base = os.environ.get("DEV_UPTRENDS_BASE_API")
        self.prod_api_base = os.environ.get("PROD_UPTRENDS_BASE_API")

        self.api_base = self.dev_api_base
        
        # Let's grab a session ID
        sess_resp = requests.get(self.api_base+f"/session?user_key={self.api_key}")
        if sess_resp.status_code != 200:
            self.session_id = "sess-placeholder"
        else:
            sess_d = sess_resp.json()
            self.session_id = sess_d["session_id"]
        
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


if __name__=="__main__":
    api = UptrendsWrapper(verbose=True)
    alerts = api.get_alert_objs(since_days_ago=7, until_days_ago=0)
    tsla_alerts = api.get_alert_objs(
        since_days_ago=60,
        until_days_ago=0,
        ticker="TSLA"
    )
    print(tsla_alerts[0].model_dump_json(indent=2, exclude={"timeline"}))

