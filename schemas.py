
from bson import ObjectId
from bson.errors import InvalidId
import datetime
from enum import Enum
import json
import numpy as np
from pydantic import BaseModel
import pytz
from typing import List, Union, Optional, Generic, TypeVar


# JSON encoder for datetime objs
class JSONEncoder_dt(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return datetime_to_isostr(obj)
        return json.JSONEncoder.default(self, obj)
    

#######################################
###  [ Pydantic Helper Functions ]  ###

def convert_to_utc_datetime(dt_obj: datetime.datetime) -> datetime.datetime:
    return dt_obj.astimezone(tz=pytz.utc)

def datetime_to_utc_date(dt_obj: datetime.datetime) -> datetime.datetime:
    # Converts datetime into UTC datetime, then clamps to date but leaves as datetime
    utc_datetime = dt_obj.astimezone(tz=pytz.utc)
    utc_date = utc_datetime.date()
    out_datetime = datetime.datetime.combine(date=utc_date, time=datetime.time(0, 0, 0, tzinfo=pytz.utc), tzinfo=pytz.utc)
    return out_datetime

def datetime_to_isostr(dt_obj: Union[datetime.datetime, datetime.date]) -> str:
    return dt_obj.strftime('%Y-%m-%dT%H:%M:%SZ')

def datestr_to_datetime(datestr: str):
    return datetime.datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%SZ").astimezone(pytz.utc)

def check_datestr(input_: Union[str, datetime.datetime]) -> datetime.datetime:
    if type(input_) is datetime.datetime:
        return input_
    else:
        return datestr_to_datetime(input_)
    

def timedelta_to_str(span_: datetime.timedelta):
    out_ms = span_.microseconds // 1000
    return str(out_ms)

class _ObjectID(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            return ObjectId(str(v))
        except InvalidId:
            raise ValueError("Not a valid ObjectId")


DType=TypeVar('DType')

class _ndarray(np.ndarray, Generic[DType]):
    """Wrapper class for numpy arrays that stores and validates type information.
    This can be used in place of a numpy array, but when used in a pydantic BaseModel
    or with pydantic.validate_arguments, its dtype will be *coerced* at runtime to the
    declared type.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, val):
        # If numpy cannot create an array with the request dtype, an error will be raised
        # and correctly bubbled up.
        np_array = np.array(val)
        return np_array


def arr_to_lst(arr: np.ndarray):
    return arr.tolist()

BASE_JSON_ENCODERS = {
    datetime.datetime: datetime_to_isostr,
    datetime.date: datetime_to_isostr,
    datetime.timedelta: timedelta_to_str,
    _ObjectID: lambda x: str(x),
    np.ndarray: arr_to_lst
}

#################################
###  [ Pydantic Base Model ]  ###

class Baseline(BaseModel):
    class Config:
        json_encoders = BASE_JSON_ENCODERS
