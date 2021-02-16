import datetime

def time_delta_seconds(ts_start, ts_end):
    if not isinstance(ts_start, datetime.datetime) or not isinstance(ts_end, datetime.datetime):
        raise ValueError('The timestamps are not of <class \'datetime.datetime\'>')
    return (ts_end - ts_start).total_seconds()

def time_add_seconds(ts, seconds):
    if not isinstance(ts, datetime.datetime) or not isinstance(seconds, float):
        raise ValueError('The timestamp is not of <class \'datetime.datetime\'> or the seconds value is not a float')
    return ts + datetime.timedelta(0,seconds)