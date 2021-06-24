import datetime
from enum import Enum

import pandas as pd
from pandas import *

from numpy import *
from dateutil.parser import parse
import dateparser
import holidays


class TimeType(Enum):
    DATE = 'date'
    DURATION = 'duration'
    NONE = 'none'


def time_encoding(df: DataFrame) -> DataFrame:
    """Entry point for time encoding
    Encodes the columns of string of the given DataFrame if they are date or duration

    :param df:
    :return:
    """
    df_output = DataFrame()
    for column_name in df.keys():
        column_value = df[column_name]
        column_type = is_time_or_duration(column_value)

        if column_type == TimeType.DURATION.value:
            df_output.append(parse_duration(column_value, column_name))
        elif column_type == TimeType.DATE.value:
            result_df, duration = parse_date(column_value, column_name)
            df_output.append(result_df)
            df_output.append(parse_duration(duration, column_name))
        else:
            df_output[column_name] = column_value

    return df_output


def is_time_or_duration(column: list):
    """Returns whether the column contains dates, durations, or otherwise

    :param column:
    :return:
    """
    column_type = TimeType.NONE.value

    if is_duration(column):
        column_type = TimeType.DURATION.value
    elif is_date(column):
        column_type = TimeType.DATE.value

    return column_type


def is_date(column: list) -> bool:
    """Returns whether all string can be interpreted as a date.

    Accepts empty string and None Object in python
    Function take from https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
    :param column: list of str, strings to check for date
    :return: True if all string of column are dates
    """
    for value in column:
        if isinstance(value, str):
            if value != "" and value != 'None':
                try:
                    float(value)
                    return False
                except ValueError:
                    try:
                        parse(value)
                    except ValueError:
                        return False
        elif isinstance(value, datetime) or value is None:
            pass
        else:
            return False

    return True


def is_duration(column: list) -> bool:
    """Returns whether all string can be interpreted as a duration.

    Accepts empty string and None Object in python
    :param column: list of str, strings to check for periods of time
    :return: True if all string of column are periods of time
    """
    for value in column:
        if isinstance(value, str):
            if value != "" and value != 'None':
                try:
                    float(value)
                    return False
                except ValueError:
                    groups = format_string_duration_parse(value)
                    if not all([
                        (len(group) == 2 and group[0].isnumeric() and group[1] in duration_allowed_word)
                        for group in groups
                    ]):
                        return False
        elif value is None:
            pass
        else:
            return False

    return True


duration_allowed_word = ['d', 'days', 'h', 'hours', 'm', 'minutes', 's', 'seconds']


def format_string_duration_parse(string: str) -> list:
    """Returns a list containing the given string split

    :param string:
    :return:
    """
    string = string.replace(" ", "")

    chars = [string[0]]
    for char in string[1:]:
        if not chars[-1].isnumeric() and char.isnumeric():
            chars += ['|']
            chars += [char]
        elif chars[-1].isnumeric() and not char.isnumeric():
            chars += ['_']
            chars += [char]
        else:
            chars += [char]
    # From 18d5h38m36s, I want have for example 18_d|5_h|38_m|36_s

    formatted_string = [tuple(group.split('_')) for group in "".join(chars).split('|')]
    # recreates the string, then splits it first to have the number_keyword and then create the tuples

    return formatted_string


def is_special_occasion(date):
    countries = ['AR', 'AU', 'AT', 'BY', 'BE', 'BR', 'BG', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'EG', 'EE', 'FI', 'FR',
               'DE', 'GR', 'HU', 'IS', 'IN', 'IE', 'IL', 'IT', 'JM', 'JP', 'LT', 'MX', 'MA', 'NL', 'NZ', 'PL', 'PT',
               'RO', 'RU', 'SA', 'RS', 'SK', 'SI', 'ZA', 'ES', 'SE', 'CH', 'TR', 'UA', 'AE', 'GB', 'US']
    for country in countries:
        holiday = holidays.CountryHoliday(country)
        if date.strftime("%m-%d-%Y") in holiday:
            return True
    return False


def encode_date(value):
    if isinstance(value, datetime):
        date = value
    else:
        date = dateparser.parse(value)  # Returns a datetime type
    return [date.isoweekday(), date.day, date.month, date.year, date.hour, date.minute, date.second,
            is_special_occasion(date)], date


def encode_dates_for_duration(date: datetime, last_date: datetime):
    tot_seconds = int((date-last_date).total_seconds())

    if tot_seconds > 0:
        tot_minutes = int(tot_seconds / 60)
        tot_hours = int(tot_minutes / 60)
        days = int(tot_hours / 24)
        return str(days) + 'd' + str(tot_hours % 24) + 'h' + str(tot_minutes % 60) + 'm' + str(tot_seconds % 60) + 's'
    else:
        return None


def parse_date(column: list, column_name: str) -> (DataFrame, list):
    """Parses strings of column into datetime objects and returns a DataFrame

    :param column: list of str, strings to parse into date
    :param column_name:
    :return:
    """
    columns = [(column_name+'_date_week_day'), (column_name+'_date_day'), (column_name+'_date_month'),
               (column_name+'_date_year'), (column_name+'_date_hours'), (column_name+'_date_minutes'),
               (column_name+'_date_seconds'), (column_name+'_date_special_occasion')]

    # encoded_dates = [
    #     [None for _ in columns]
    #     if (value is None or value == '' or value == 'None')
    #     else encode_date(value)
    #     for value in column
    # ]
    encoded_dates = []
    elapsed_times = []
    last_date = None

    for value in column:
        if value is None or value == '' or value == 'None':
            encoded_dates += [[None for _ in columns]]
            elapsed_times += [None]
        else:
            listed_date, date = encode_date(value)
            encoded_dates += [listed_date]
            if last_date is None:
                elapsed_times += [None]
            else:
                elapsed_times += [encode_dates_for_duration(date, last_date)]
            last_date = date

    results_df = DataFrame(data=encoded_dates, columns=columns)
    results_df = results_df.where(pd.notnull(results_df), None)

    return results_df, elapsed_times


def encode_duration(value, columns):
    formatted_string = format_string_duration_parse(value)
    row = [0 for _ in columns]

    for group in formatted_string:
        if group[1] == duration_allowed_word[0] or group[1] == duration_allowed_word[1]:
            row[0] += int(group[0])
        elif group[1] == duration_allowed_word[2] or group[1] == duration_allowed_word[3]:
            row[1] += int(group[0])
        elif group[1] == duration_allowed_word[4] or group[1] == duration_allowed_word[5]:
            row[2] += int(group[0])
        elif group[1] == duration_allowed_word[6] or group[1] == duration_allowed_word[7]:
            row[3] += int(group[0])

    return row


def parse_duration(column: list, column_name: str) -> DataFrame:
    """Parses strings of column into datetime objects and returns a DataFrame

    I assume that I receive the duration in one of the following format
    - number (milliseconds)
    - number d number h number m number
    - number days number hours number minutes number seconds
    - number days

    All space will be removed
    :param column:
    :param column_name:
    :return:
    """
    columns = [(column_name+'_elapsed_days'), (column_name+'_elapsed_hours'), (column_name+'_elapsed_minutes'),
               (column_name+'_elapsed_seconds')]

    encoded_durations = [
        encode_duration(value, columns)
        if (isinstance(value, str) and value != '' and value != 'None')
        else [None for _ in columns]
        for value in column
    ]

    results_df = DataFrame(data=encoded_durations, columns=columns)
    results_df = results_df.where(pd.notnull(results_df), None)

    return results_df


if __name__ == '__main__':
    time_test = [
        '1990-12-1',
        '',
        None,
        'None',
        '01/19/1990',
        '01/19/90',
        'Jan 1990',
        'January1990',
        '2005/3',
        'Monday at 12:01am',
        'January 1, 2047 at 8:21:00AM',
    ]

    duration_test = [
        '2d9h32m46s',
        '2d 9h',
        '',
        None,
        'None',
        '2days9hours37minutes46seconds',
        '2days 9hours 37minutes 46seconds',
    ]

    print(is_time_or_duration(time_test))
    print(is_time_or_duration(duration_test))

    parsed_dates, durations = parse_date(time_test, 't1')
    print(parsed_dates.head())
    print(durations)
    print(parse_duration(durations, 't1').head())
    print(parse_duration(duration_test, 't2').head())
