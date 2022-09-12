from datetime import datetime, date, time, timedelta
import pytz
import math

holidays = [date(2010, 1, 1), date(2010, 1, 18), date(2010, 2, 15),
            date(2010, 4, 2), date(2010, 5, 31), date(2010, 7, 5), date(2010, 9, 6),
            date(2010, 11, 25), date(2010, 12, 24), date(2010, 12, 31),

            date(2011, 1, 17), date(2011, 2, 21), date(2011, 4, 22), date(2011, 5, 30),
            date(2011, 7, 4), date(2011, 9, 5), date(2011, 11, 24),
            date(2011, 12, 26),

            date(2012, 1, 2), date(2012, 1, 16), date(2012, 2, 20), date(2012, 4, 6), date(2012, 5, 28),
            date(2012, 7, 4), date(2012, 9, 3),
            date(2012, 11, 22), date(2012, 12, 25),

            date(2013, 1, 1), date(2013, 1, 21), date(2013, 2, 18), date(2013, 3, 29), date(2013, 5, 27), date(2013, 7, 4),
            date(2013, 9, 2), date(2013, 11, 28), date(2013, 12, 25),

            date(2014, 1, 1), date(2014, 1, 20), date(2014, 2, 17), date(2014, 4, 18), date(2014, 5, 26), date(2014, 7, 4),
            date(2014, 9, 1), date(2014, 11, 27), date(2014, 12, 25),

            date(2015, 1, 1), date(2015, 1, 19), date(2015, 2, 16), date(2015, 4, 3), date(2015, 5, 25), date(2015, 7, 3),
            date(2015, 9, 7), date(2015, 11, 26), date(2015, 12, 25),

            date(2016, 1, 1), date(2016, 1, 18), date(2016, 2, 15), date(2016, 3, 25), date(2016, 5, 30), date(2016, 7, 4),
            date(2016, 9, 5), date(2016, 11, 24), date(2016, 12, 26),

            date(2017, 1, 2), date(2017, 1, 16), date(2017, 2, 20), date(2017, 5, 29),
            date(2017, 7, 4), date(2017, 9, 4), date(2017, 11, 23), date(2017, 12, 25),

            date(2018, 1, 1), date(2018, 1, 15), date(2018, 2, 19), date(2018, 3, 30), date(2018, 5, 28), date(2018, 7, 4),
            date(2018, 9, 3), date(2018, 11, 22), date(2018, 12, 25),

            date(2019, 1, 1), date(2019, 1, 21), date(2019, 2, 18), date(2019, 4, 19), date(2019, 5, 27), date(2019, 7, 4),
            date(2019, 9, 2), date(2019, 11, 28), date(2019, 12, 25),

            date(2020, 1, 1), date(2020, 1, 20), date(2020, 2, 17), date(2020, 4, 10), date(2020, 5, 25), date(2020, 7, 3),
            date(2020, 9, 7), date(2020, 11, 26), date(2020, 12, 25),

            date(2021, 1, 1), date(2021, 1, 18), date(2021, 2, 15), date(2021, 4, 2), date(2021, 5, 31),
            date(2021, 7, 5), date(2021, 9, 6), date(2021, 11, 25), date(2021, 12, 24), date(2021, 12, 31),

            date(2022, 1, 17), date(2022, 2, 21), date(2022, 4, 15), date(2022, 5, 30),
            date(2022, 7, 4), date(2022, 9, 5), date(2022, 11, 24), date(2022, 12, 26)
            ]


def is_business_day(input_date) -> bool:
    return input_date.isoweekday() < 6 and input_date not in holidays


def get_last_session_date(end_of_session):
    now = real_market_now()
    if now.time() > end_of_session:
        now = now + timedelta(days=1)

    return get_prev_business_day(now)


def get_prev_business_day(input_date) -> date:
    if isinstance(input_date, datetime):
        src = input_date
    elif isinstance(input_date, date):
        src = datetime.combine(input_date, time(0, 0, 0))
    else:
        raise Exception("unknown input type for get_prev_business_day")

    result = src - timedelta(days=1)
    if result.isoweekday() == 7:
        result = result - timedelta(days=2)
    if result.isoweekday() == 6:
        result = result - timedelta(days=1)

    result_date = result.date()

    if result_date in holidays:
        result_date = get_prev_business_day(result_date)

    return result_date


def round_time(tm, round_to):
    begin_of_day = tm.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = (tm - begin_of_day).seconds
    rounded = math.floor(seconds / round_to) * round_to
    return (begin_of_day + timedelta(seconds=rounded)).replace(microsecond=0)


def is_rth_bar(bar_time: datetime):
    tm = bar_time.time()
    return time(9, 30) < tm <= time(16, 0)


def real_market_now() -> datetime:
    return datetime.now(pytz.timezone('US/Eastern')).replace(tzinfo=None)


def check_if_date_is_eod_date(input_date) -> date:
    last_session_date = get_last_session_date(time(16, 0))
    if last_session_date < input_date:
        date_to_output = last_session_date
    else:
        date_to_output = input_date
    if is_business_day(date_to_output):
        return date_to_output
    else:
        return get_prev_business_day(date_to_output)