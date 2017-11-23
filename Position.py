from Utils import *
from _decimal import Decimal
import math
from builtins import Exception


class Position(object):
    # trade = namedtuple('Trade',
    # ['signal','ret','cashret','capital','cost','direction','prices'])
    # return_status = namedtuple('Return',
    # ['max','max_freq','min','min_freq','ret','freq','reason','max_hold'])

    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.name = self.raw_data.name
        self.cash_return = 0
        self.close_freq = 0
        self.close_reason = ""
        self.cost = 0
        self.__opened = False
        self.closed = False
        self.curr_freq = 0
        self.max_return = 0
        self.max_return_cash_value = Decimal("0")
        self.max_return_freq = 0
        self.max_draw = Decimal("0")
        self.max_draw_cash_value = Decimal("0")
        self.max_draw_freq = 0
        self.max_draw_from_previous_high = 0
        self.max_draw_from_previous_high_freq = 0
        self.__curr_return = 0
        self.start_time = None
        self.start_price = None
        self.shares = 0
        self.initial_capital = None
        self.__curr_time = None
        self.curr_price = None
        self.curr_value = 0

    def open_trade(self, cost, shares, time):
        if self.__opened or self.closed:
            raise Exception("should not be here")
        self.cost = cost
        self.start_time = time
        self.start_price = perc_to_decimal(self.raw_data.prices[self.curr_freq])
        self.shares = shares
        self.initial_capital = self.start_price * self.shares
        self.__opened = True
        self.curr_value = self.initial_capital

    def pass_a_freq(self, time):
        if self.closed:
            raise Exception("should not be here")
        self.__curr_time = time
        self.curr_freq += 1
        self.curr_price = perc_to_decimal(self.raw_data.prices[self.curr_freq])
        self.__curr_return = (self.curr_price / self.start_price - 1) * self.raw_data.hyper_parameter.direction
        self.curr_value = self.initial_capital + (self.curr_price - self.start_price) * self.raw_data.hyper_parameter.direction * self.shares
        if self.__curr_return >= self.max_return:
            self.max_return = self.__curr_return
            self.max_return_cash_value = self.curr_value - self.initial_capital
            assert self.max_return >= 0
            assert self.max_return_cash_value >= 0
            self.max_return_freq = self.curr_freq
        if self.__curr_return <= self.max_draw:
            self.max_draw = self.__curr_return
            self.max_draw_cash_value = self.curr_value - self.initial_capital
            assert self.max_draw <= 0
            assert self.max_draw_cash_value <= 0
            self.max_draw_freq = self.curr_freq
        curr_draw_from_previous_high = self.__curr_return - self.max_return
        if curr_draw_from_previous_high <= self.max_draw_from_previous_high:
            self.max_draw_from_previous_high = curr_draw_from_previous_high
            assert self.max_draw_from_previous_high <= 0
            self.max_draw_from_previous_high_freq = self.curr_freq

    def should_stop(self):
        vol = -perc_to_float(self.raw_data.vola) / 100 / math.sqrt(252) * self.raw_data.hyper_parameter.std

        if self.max_return_freq != self.curr_freq and self.__curr_return <= self.stop_loss_perc(self.max_return, self.raw_data.hyper_parameter.maxlossforce, self.raw_data.hyper_parameter.maxloss, vol, self.raw_data.hyper_parameter.stdmul, self.curr_freq - self.max_return_freq):
            if self.__curr_return >= 0:
                return True, "Win"
            else:
                return True, "Loss"
        if self.curr_freq >= self.raw_data.hyper_parameter.maxholding:
            return True, "Hold"

        return False, None

    @staticmethod
    def vol_mul(vol, stdmul, d):
        if stdmul == 0:
            volmul = float("-inf")
        elif stdmul == 1:
            volmul = vol
        elif stdmul == 2:
            volmul = math.sqrt(d) * vol
        elif stdmul == 3:
            volmul = d * vol
        else:
            raise Exception("should not be here")
        return volmul

    def stop_loss_perc(self, maxret, maxlossforce, maxloss, volcrit, stdmul, d):
        return max(maxlossforce, float(maxret) + max(self.vol_mul(volcrit, stdmul, d), maxloss))

    def close_trade(self, cost, close_reason):
        if (not self.__opened) or self.closed:
            raise Exception("should not be here")
        self.cash_return = (self.curr_price - self.start_price) * self.raw_data.hyper_parameter.direction * self.shares
        self.cost += cost
        self.close_freq = self.curr_freq
        self.close_reason = close_reason
        self.closed = True
        self.__opened = False
        return self.cash_return + self.initial_capital - cost

    def to_csv(self, writer):
        writer.writerow(
            self.raw_data.signal_to_tuple() + (
            self.max_return, self.max_return_freq, self.max_draw, self.max_draw_freq, self.max_draw_from_previous_high, self.max_draw_from_previous_high_freq, self.__curr_return, self.curr_freq, self.close_reason, self.cash_return, self.initial_capital, self.cost,
            self.raw_data.hyper_parameter.direction) + tuple(self.raw_data.prices[:]))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.raw_data.name}({self.start_time},{self.start_price},{self.shares})"
