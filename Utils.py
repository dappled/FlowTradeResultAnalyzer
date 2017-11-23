from _decimal import Decimal

import math


def perc_to_float(input_perc):
    return float(str.rstrip(input_perc, "%"))


def perc_to_decimal(input):
    if not input:
        return Decimal("nan")
    else:
        return Decimal(str.rstrip(input, "%"))

def is_holiday(date):
    month = date.month
    year = date.year
    day = date.day
    if ((month == 12 and day == 25 and year == 2013) or (month == 1 and day == 1 and year == 2014) or (month == 1 and day == 20 and year == 2014) or (month == 2 and day == 17 and year == 2014) or (month == 4 and day == 18 and year == 2014) or (
                        month == 5 and day == 26 and year == 2014) or (month == 7 and day == 4 and year == 2014) or (month == 9 and day == 1 and year == 2014) or (month == 11 and day == 27 and year == 2014) or (
                        month == 12 and day == 25 and year == 2015) or (month == 1 and day == 1 and year == 2015) or (month == 1 and day == 19 and year == 2015) or (month == 2 and day == 16 and year == 2015) or (
                        month == 4 and day == 3 and year == 2015) or (
                        month == 5 and day == 25 and year == 2015) or (month == 7 and day == 3 and year == 2015) or (month == 9 and day == 7 and year == 2015) or (month == 11 and day == 26 and year == 2015) or (
                        month == 12 and day == 25 and year == 2015) or (month == 1 and day == 1 and year == 2016) or (month == 1 and day == 18 and year == 2016) or (month == 2 and day == 15 and year == 2016) or (
                        month == 3 and day == 25 and year == 2016) or (month == 5 and day == 30 and year == 2016) or (month == 7 and day == 4 and year == 2016) or (month == 9 and day == 5 and year == 2016) or (
                        month == 11 and day == 24 and year == 2016) or (
                        month == 12 and day == 26 and year == 2016) or (month == 1 and day == 2 and year == 2017) or (month == 1 and day == 16 and year == 2017) or (month == 2 and day == 20 and year == 2017) or (
                        month == 4 and day == 14 and year == 2017) or (month == 5 and day == 29 and year == 2017) or (month == 7 and day == 4 and year == 2017) or (month == 9 and day == 4 and year == 2017) or (
                            month == 11 and day == 23 and year == 2017 or (month == 12 and day == 25 and year == 2017))):
        return True
    return False
