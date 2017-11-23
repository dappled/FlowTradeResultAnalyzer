import collections
import csv
import datetime
import numpy as np
from builtins import list, float, sorted, Exception

import pandas as pd
from pandas.tseries.offsets import BDay

from Position import Position
from Utils import *

import matplotlib.pyplot as plt
import math


def shares_considier_fee(cash, price):
    return min(math.floor((float(cash) - 1) / (float(price) + 0.02)), math.floor(float(cash) / (float(price) + 0.025)))


class Portfolio(object):
    # initial_capital = 1000000
    # borrow = -200000
    # min_capital_single_trade = 100
    # max_capital_single_trade = 50000
    # max_capital_single_name = 100000
    # unit_capital_single_trade = 10000
    # max_trade_single_name = 3
    # borrow_cost_annual = Decimal(0.02)
    # max_uvol_weight = 2.5
    # max_uoi_weight = 2.5

    def __init__(self, trades, config):
        self.config = config

        self.trades = collections.OrderedDict()
        self.separate_trades(trades)
        self.holdings = {}
        self.capital = self.config.portfolio.initial_capital
        self.total_cash_return = 0
        self.total_tran_cost = 0
        self.total_borrow_cost = 0
        self.current_freq = None
        self.finished_trades = []
        self.daily_status = []
        self.current_holding_cash = 0

        if self.config.log['tradef'] is not None:
            with open(rf"{self.config.log['directory']}\{self.config.log['tradef']}", 'w', newline='') as csvfile:
                fieldsname = (
                    'date', 'name', 'vol', 'adv', 'oi', 'uvol', 'uoi', 'vd', 'iv', 'ivchg', 'edate', 'conf', 'rate', 'vola', 'cap', 'unadjprice', 'max', 'freq', 'min', 'freq', 'draw from high', 'freq', 'ret', 'freq', 'reason',
                    'cashret', 'capital', 'cost', 'direction',
                    'price') + tuple(f'{i}' for i in np.arange(1, 41))
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(fieldsname)

    def separate_trades(self, trades):
        yesterday_trades = []

        prev_date = None
        for trade in trades:
            current_freq = datetime.datetime.strptime(trade.date, "%m/%d/%Y").date()
            if prev_date is not None and prev_date != current_freq:
                if prev_date in self.trades:
                    raise Exception("should not be here")
                self.trades[prev_date] = yesterday_trades
                yesterday_trades = []

            yesterday_trades.append(trade)
            prev_date = current_freq
        if len(yesterday_trades) == 0:
            raise Exception("should not be here")
        if prev_date in self.trades:
            raise Exception("should not be here")
        self.trades[prev_date] = yesterday_trades

    def run(self, start_freq, end_freq):
        try:
            self.current_freq = start_freq
            while self.current_freq <= end_freq:
                python_time = pd.to_datetime(self.current_freq).date()
                if python_time in self.trades.keys():
                    today_trades = self.trades[python_time]
                else:
                    today_trades = []

                self.pass_a_freq(today_trades, self.current_freq == end_freq)

                # print("passed {}".format(python_time))

                self.current_freq += BDay(1)
                while is_holiday(self.current_freq):
                    self.current_freq += BDay(1)

            if self.config.detail_report:
                return self.detail_report(True)
            else:
                return self.simple_report(True)
        except Exception:
            if self.config.detail_report:
                return self.detail_report(False)
            else:
                return self.simple_report(False)

    def simple_report(self, valid):
        if not valid:
            return Decimal("-inf")

        total_return = self.total_cash_return - self.total_borrow_cost - self.total_tran_cost
        if round(self.capital, 2) != round(self.config.portfolio.initial_capital + total_return, 2):
            raise Exception("should not be here")
        return total_return

    def detail_report(self, valid):
        if not valid:
            return None

        return self.analyze_return()

    def analyze_return(self):
        total_net_profit = self.total_cash_return - self.total_borrow_cost - self.total_tran_cost
        if not Decimal.is_infinite(self.capital) and round(self.capital, 2) != round(self.config.portfolio.initial_capital + total_net_profit, 2):
            raise Exception("should not be here")
        annualized_return = float('nan') if Decimal.is_infinite(self.config.portfolio.initial_capital) else math.pow(total_net_profit / self.config.portfolio.initial_capital + 1, 0.5) - 1

        gross_profit = 0
        net_profit = 0
        total_win_perc = 0
        gross_loss = 0
        net_loss = 0
        total_loss_perc = 0
        sq_sum_net_profit = 0
        sq_sum_net_loss = 0
        nwin = 0
        nloss = 0
        nall = 0
        sumalltrade = 0
        largest_winning_trade = Decimal("-inf")
        largest_losing_trade = Decimal("inf")
        total_time_trades = 0
        total_time_winning_trades = 0
        total_time_losing_trades = 0

        # trade related statistics
        for trade in self.finished_trades:
            if not trade.closed:
                raise Exception("should not be here")
            ret = float(trade.cash_return - trade.cost)
            perc = ret / float(trade.initial_capital)
            if trade.cash_return >= 0:
                nwin += 1
                gross_profit += trade.cash_return
                net_profit += ret
                total_win_perc += perc
                sq_sum_net_profit += ret ** 2
                total_time_winning_trades += trade.close_freq
            else:
                nloss += 1
                gross_loss += trade.cash_return
                net_loss += ret
                total_loss_perc += perc
                sq_sum_net_loss += ret ** 2
                total_time_losing_trades += trade.close_freq
            total_time_trades += trade.close_freq

            if trade.cash_return < largest_losing_trade:
                largest_losing_trade = trade.cash_return
            if trade.cash_return > largest_winning_trade:
                largest_winning_trade = trade.cash_return

            nall += 1
            sumalltrade += trade.cash_return

        profit_factor = abs(gross_profit / gross_loss)
        hitratio = nwin / float(nall)
        avg_trade_net_profit = (sumalltrade - self.total_tran_cost) / nall
        avg_winning_trade = net_profit / nwin
        std_winning_trade = (sq_sum_net_profit / nwin - avg_winning_trade ** 2) ** 0.5
        avg_winning_trade_perc = total_win_perc / nwin
        avg_losing_trade = net_loss / nloss
        std_losing_trade = (sq_sum_net_loss / nloss - avg_losing_trade ** 2) ** 0.5
        avg_losing_trade_perc = total_loss_perc / nloss
        ratio_avg_win_avg_loss = abs(avg_winning_trade / avg_losing_trade)
        avg_time_in_total_trades = total_time_trades / nall
        avg_time_in_winning_trades = total_time_winning_trades / nwin
        avg_time_in_losing_trades = total_time_losing_trades / nloss

        # daily capital related statistics
        default_status = [datetime.date(1970, 1, 1), 0, 0]
        max_drawdown = default_status
        max_drawdown_tmp = default_status
        drop_from_last_high = 0
        drop_from_last_high_tmp = 0
        previous_high = default_status
        previous_high_tmp = default_status
        status_count = 0
        status_len = len(self.daily_status)
        underwater_perc = []

        sum_daily_capital_used = 0
        sum_daily_trade_hold = 0
        max_daily_capital_used = 0
        max_daily_capital_used_trade_hold = 0
        date_max_daily_capital_used = None

        sum_ret = 0
        sq_sum = 0
        sq_sum_minus_rfr = 0
        sq_sum_lowerhalf_minus_target = 0
        last_freq_cash_value = 0
        for status in self.daily_status:
            status_count += 1
            cash_value = status[1]

            capital_used = status[3]
            trade_hold = status[4]
            sum_daily_capital_used += capital_used
            sum_daily_trade_hold += trade_hold
            if capital_used > max_daily_capital_used:
                max_daily_capital_used = capital_used
                max_daily_capital_used_trade_hold = trade_hold
                date_max_daily_capital_used = status[0]

            # draw down
            if cash_value >= previous_high_tmp[1] or status_count == status_len:
                if drop_from_last_high_tmp <= drop_from_last_high:
                    max_drawdown = max_drawdown_tmp
                    previous_high = previous_high_tmp
                    drop_from_last_high = drop_from_last_high_tmp

                drop_from_last_high_tmp = 0
                previous_high_tmp = status
                max_drawdown_tmp = default_status

                if status_count == status_len and cash_value < previous_high_tmp[1]:
                    underwater_perc.append((cash_value - previous_high_tmp[1]) / previous_high_tmp[1])
                else:
                    underwater_perc.append(0)
            else:
                diff = cash_value - previous_high_tmp[1]
                if diff < drop_from_last_high_tmp:
                    drop_from_last_high_tmp = diff
                    max_drawdown_tmp = status

                underwater_perc.append(diff / previous_high_tmp[1])

            # ratios
            if last_freq_cash_value != 0:
                ret = float(cash_value / last_freq_cash_value) - 1
                sum_ret += ret
                sq_sum += ret ** 2
                sq_sum_minus_rfr += (ret - self.config.portfolio.interest_rate) ** 2
                sq_sum_lowerhalf_minus_target += min(0, ret - self.config.portfolio.target_rate) ** 2

            last_freq_cash_value = cash_value

        max_drawdown_value = max_drawdown[1] - previous_high[1]

        mean_daily_return = sum_ret / status_len
        std_daily_return = (sq_sum / status_len - mean_daily_return ** 2) ** 0.5

        mean_minus_rfr_daily = mean_daily_return - self.config.portfolio.interest_rate
        std_minus_rfr_daily = (sq_sum_minus_rfr / status_len - mean_minus_rfr_daily ** 2) ** 0.5

        sharpeRatio = mean_minus_rfr_daily / std_minus_rfr_daily * (255 ** 0.5)

        mean_minus_target_daily = mean_daily_return - self.config.portfolio.target_rate
        std_lowerhalf_minus_target_daily = (sq_sum_lowerhalf_minus_target / status_len) ** 0.5

        sortinoRatio = mean_minus_target_daily / std_lowerhalf_minus_target_daily / (2 ** 0.5) * (255 ** 0.5)

        avg_daily_capital_used = sum_daily_capital_used / status_len
        avg_daily_trade_hold = sum_daily_trade_hold / status_len

        # std_trades_return = np.std([trade.cash_return - trade.cost for trade in self.finished_trades])
        # no_outlier_trades = [trade for trade in self.finished_trades if (trade.cash_return - trade.cost) <= avg_trade_net_profit + std_trades_return * 3 and (trade.cash_return - trade.cost) >= avg_trade_net_profit - std_trades_return * 3]
        # select_net_profit = sum([trade.cash_return - trade.cost for trade in no_outlier_trades])
        # avg_drawdown =

        # if self.config.output_file is not None:
        #     with open(self.config.output_file, 'a', newline='') as csvfile:
        #         writer = csv.writer(csvfile, delimiter=',')
        #         writer.writerow((total_net_profit, annualized_return, self.total_cash_return, self.total_tran_cost, self.total_borrow_cost))
        #         writer.writerow(statistics)

        statistics = pd.DataFrame.from_items([('Total Net Profit', [total_net_profit]), ('Annualized Net Profit', [annualized_return]), ('Net profit divided by dd', [total_net_profit / -max_drawdown_value]),
                                              ('Gross Profit', [gross_profit]), ('Gross Loss', [gross_loss]),
                                              ('Profit Factor', [profit_factor]), ('Total Number of Trades', [nall]), ('Percent Profitable', [hitratio]), ('Winning Trades', [nwin]), ('Losing Trades', [nloss]),
                                              ('Avg. Trade Net Profit', [avg_trade_net_profit]), ('Avg. Winning Trades', [avg_winning_trade]), ('Std. Winning Trades', [std_winning_trade]),
                                              ('Avg. Winning Trade Perc', [avg_winning_trade_perc]), ('Avg. Losing Trade', [avg_losing_trade]), ('Std. Losing Trade', [std_losing_trade]),('Avg. Losing Trade Perc',[avg_losing_trade_perc]),
                                              ('Ratio Avg. Win:Avg. Loss', [ratio_avg_win_avg_loss]), ('Largest Winning Trade', [largest_winning_trade]), ('Largest Losing Trade', [largest_losing_trade]),
                                              ('Avg Daily Capital Usage', [avg_daily_capital_used]), ('Avg Daily Position Hold', [avg_daily_trade_hold]), ('Max Daily Capital Usage', [max_daily_capital_used]),
                                              ('Position Hold', [max_daily_capital_used_trade_hold]), ('Date of Max Daily Capital Usage', [date_max_daily_capital_used]),
                                              ('Avg Time In Total Trades', [avg_time_in_total_trades]), ('Avg Time In Winning Trade', [avg_time_in_winning_trades]), ('Avg Time In Losing Trade', [avg_time_in_losing_trades]),
                                              ('Max Drawdown', [max_drawdown_value]), ('Max Drawdown to capital used', [-max_drawdown_value / avg_daily_capital_used]), ('Date of Max Drawdown', [max_drawdown[0]]), ('Date of Max Drawdown Previous High', [previous_high[0]]),
                                              ('Mean Daily Return', [mean_daily_return]), ('Std Daily Return', [std_daily_return]), ('Sharpe Ratio', [sharpeRatio]), ('Sortino Ratio', [sortinoRatio])],
                                             orient='index', columns=[self.config.name])

        # saving
        # statistics
        if self.config.log['statsf'] is not None:
            statistics.to_csv(rf"{self.config.log['directory']}\{self.config.log['statsf']}")
        # equity line
        if self.config.log['equityg'] is not None:
            xs = [x[0] for x in self.daily_status]
            ys = [x[1] - self.config.portfolio.initial_capital for x in self.daily_status]
            plt.plot(xs, ys)
            plt.savefig(rf"{self.config.log['directory']}\{self.config.log['equityg']}")
            plt.clf()
        # underwater
        if self.config.log['underwaterg'] is not None:
            xs = [x[0] for x in self.daily_status]
            ys = underwater_perc
            plt.plot(xs, ys)
            plt.savefig(rf"{self.config.log['directory']}\{self.config.log['underwaterg']}")
            plt.clf()
        # capital used
        if self.config.log['capitalg'] is not None:
            xs = [x[0] for x in self.daily_status]
            ys = [x[2] for x in self.daily_status]
            plt.plot(xs, ys)
            plt.savefig(rf"{self.config.log['directory']}\{self.config.log['capitalg']}")
            plt.clf()

        winning_trade, losing_trade = self.separate_winning_losing_trade()
        # mae
        if self.config.log['maeg'] is not None:
            winning_ret_perc = [tr.cash_return / tr.initial_capital for tr in winning_trade]
            winning_draw_perc = [abs(tr.max_draw) for tr in winning_trade]
            losing_ret_perc = [abs(tr.cash_return) / tr.initial_capital for tr in losing_trade]
            losing_draw_perc = [abs(tr.max_draw) for tr in losing_trade]
            assert max(tr.max_draw for tr in winning_trade) <= 0
            assert max(tr.max_draw for tr in losing_trade) <= 0
            assert max(tr.max_draw_cash_value for tr in winning_trade) <= 0
            assert max(tr.max_draw_cash_value for tr in losing_trade) <= 0
            assert all(tr.cash_return < 0 for tr in losing_trade)
            assert all(tr.cash_return >= 0 for tr in winning_trade)
            figmae = plt.figure()
            axmae = figmae.add_subplot(111)
            axmae.scatter(winning_draw_perc,winning_ret_perc,c='g',s=25,alpha=0.4,marker='^')
            axmae.scatter(losing_draw_perc,losing_ret_perc,c='r',s=25,alpha=0.4,marker='v')
            figmae.savefig(rf"{self.config.log['directory']}\{self.config.log['maeg']}.png")
            plt.clf()
        # mte
        if self.config.log['mteg'] is not None:
            winning_ret_perc = [tr.cash_return / tr.initial_capital for tr in winning_trade]
            winning_draw_from_high_perc = [abs(tr.max_draw_from_previous_high) for tr in winning_trade]
            losing_ret_perc = [abs(tr.cash_return) / tr.initial_capital for tr in losing_trade]
            losing_draw_from_high_perc = [abs(tr.max_draw_from_previous_high) for tr in losing_trade]
            assert max(tr.max_draw_from_previous_high for tr in winning_trade) <= 0
            assert max(tr.max_draw_from_previous_high for tr in losing_trade) <= 0
            figmae = plt.figure()
            axmae = figmae.add_subplot(111)
            axmae.scatter(winning_draw_from_high_perc,winning_ret_perc,c='g',s=25,alpha=0.4,marker='^')
            axmae.scatter(losing_draw_from_high_perc,losing_ret_perc,c='r',s=25,alpha=0.4,marker='v')
            figmae.savefig(rf"{self.config.log['directory']}\{self.config.log['mteg']}.png")
            plt.clf()
        # mfe
        if self.config.log['mfeg'] is not None:
            winning_ret_perc = [tr.cash_return / tr.initial_capital for tr in winning_trade]
            winning_runup_perc = [abs(tr.max_return) for tr in winning_trade]
            losing_ret_perc = [abs(tr.cash_return) / tr.initial_capital for tr in losing_trade]
            losing_runup_perc = [abs(tr.max_return) for tr in losing_trade]
            assert max(tr.max_return_cash_value for tr in winning_trade) >= 0
            assert max(tr.max_return_cash_value for tr in losing_trade) >= 0
            figmfe = plt.figure()
            axmfe = figmfe.add_subplot(111)
            axmfe.scatter(winning_runup_perc, winning_ret_perc, c='g', s=25, alpha=0.4, marker='^')
            axmfe.scatter(losing_runup_perc, losing_ret_perc, c='r', s=25, alpha=0.4, marker='v')
            figmfe.savefig(rf"{self.config.log['directory']}\{self.config.log['mfeg']}.png")
            plt.clf()

        return statistics
        # return f"{nall},{hitratio:.2p},{sumalltrade / nall:.2p},{sumwintrade/nwin:.2p},{sumlosetrade/nlose:.2p}"

    def separate_winning_losing_trade(self):
        winning_trade = []
        losing_trade = []
        for trade in self.finished_trades:
            if not trade.closed:
                raise Exception("should not be here")
            ret = trade.cash_return
            if ret >= 0:
                winning_trade.append(trade)
            else:
                losing_trade.append(trade)
        return winning_trade,losing_trade

    def pass_a_freq(self, trades, liquidate):
        self.current_holding_cash = 0
        for name, positions in self.holdings.items():
            for position in positions:
                position.pass_a_freq(self.current_freq)
                self.current_holding_cash += position.curr_value

        if liquidate:
            self.liquidate_positions()
        else:
            self.close_positions()

        clean_trade = sorted(trades, key=lambda td: self.edge_no_limit(td), reverse=True)
        for trade in clean_trade:
            self.add_position(trade)

        self.pay_borrow_cost()

        capital_used = 0
        trade_hold = 0
        for name, positions in self.holdings.items():
            for position in positions:
                capital_used += position.initial_capital
                trade_hold += 1

        if liquidate and self.current_holding_cash != 0:
            raise Exception("should not be here")

        self.daily_status.append((self.current_freq, self.capital + self.current_holding_cash, self.capital, capital_used, trade_hold))

    def add_position(self, trade):
        accept = trade.have_valid_prices()
        name = trade.name

        if name in self.holdings.keys():
            holdings = self.holdings[name]
            if len(holdings) >= self.config.portfolio.max_trade_per_name:
                accept = False
            else:
                total_holdings = sum(position.initial_capital for position in holdings)
                if total_holdings > self.config.portfolio.max_cash_per_name:
                    accept = False

        if accept:
            capital_want = self.capital_wanted(trade)
            capital_allow = self.capital + self.config.portfolio.allow_borrow

            while capital_want > capital_allow:
                self.remove_least_profitable_position("capital({})".format(name))
                capital_allow = self.capital + self.config.portfolio.allow_borrow

            start_price = Decimal(trade.prices[0])
            shares = shares_considier_fee(capital_want, start_price)
            if shares < 0:
                raise Exception("should not be here")
            cost = self.tran_cost(shares, start_price) + self.slippage(shares) if self.config.portfolio.care_transaction_fee else 0

            new_position = Position(trade)
            new_position.open_trade(cost, shares, self.current_freq)

            self.change_capital(-(new_position.initial_capital + cost))
            self.total_tran_cost += cost

            if name not in self.holdings:
                self.holdings[name] = []
            self.holdings[name].append(new_position)
            self.current_holding_cash += new_position.curr_value

    def close_positions(self):
        for name in self.holdings.keys():
            hold = self.holdings[name]
            for position in hold:
                should_stop, reason = position.should_stop()
                if should_stop:
                    self.close_position(position, reason)

            hold[:] = [_ for _ in hold if not _.closed]
        keys = [k for k, v in self.holdings.items() if len(v) == 0]
        for k in keys:
            del self.holdings[k]

    def liquidate_positions(self):
        for name in self.holdings.keys():
            holdings = self.holdings[name]
            for position in holdings:
                self.close_position(position, "force")
            if len([_ for _ in holdings if not _.closed]) > 0:
                raise Exception("should not be here")
            holdings.clear()
        self.holdings.clear()

    def pay_borrow_cost(self):
        if self.capital < 0:
            cost = abs(self.capital) * self.config.portfolio.borrow_rate_daily
            while self.capital - cost < -self.config.portfolio.allow_borrow:
                self.remove_least_profitable_position("borrorw")
            self.change_capital(-cost)
            self.total_borrow_cost += cost

    def close_position(self, position, reason):
        if position.curr_freq == 0:
            raise Exception("should not be here")
        exit_price = position.curr_price
        cost = self.tran_cost(position.shares, exit_price) + self.slippage(position.shares) if self.config.portfolio.care_transaction_fee else 0
        capital_change = position.close_trade(cost, reason)

        self.total_cash_return += position.cash_return
        self.total_tran_cost += cost
        self.change_capital(capital_change)
        self.current_holding_cash -= position.curr_value

        self.write_trade(position)
        self.finished_trades.append(position)

    def change_capital(self, change):
        self.capital += change
        if self.capital < -self.config.portfolio.allow_borrow:
            raise Exception("should not be here")

    def remove_least_profitable_position(self, reason):
        maxholding = max(max(self.proportion_day_till_max(position) for position in holdings) for holdings in self.holdings.values())
        if maxholding == 0:
            raise Exception("capital")
        current_return = Decimal("inf")
        toremove = None
        for name in self.holdings.keys():
            holdings = self.holdings[name]
            for position in holdings:
                if self.proportion_day_till_max(position) == maxholding:
                    if position.cash_return < current_return:
                        toremove = position
                        current_return = position.cash_return
        if toremove is None:
            raise Exception("should not be here")
        remove_holding = self.holdings[toremove.name]
        list.remove(remove_holding, toremove)
        if len(remove_holding) == 0:
            del self.holdings[toremove.name]
        self.close_position(toremove, reason)

    @staticmethod
    def proportion_day_till_max(position):
        return position.curr_freq / position.raw_data.hyper_parameter.maxholding

    def capital_wanted(self, trade):
        return min(self.config.portfolio.max_cash_per_trade,
                   max(Decimal(self.edge(trade)) * self.config.portfolio.unit_cash_per_trade, self.config.portfolio.min_cash_per_trade))

    def edge(self, trade):
        uvol_weight = min(self.config.portfolio.max_uvol_weight, max(0, math.sqrt(float(trade.uvol) - trade.hyper_parameter.minuvol)))
        uoi_weight = min(self.config.portfolio.max_uoi_weight, max(0, math.sqrt(float(trade.uoi) - trade.hyper_parameter.minuoi)))

        return uvol_weight + uoi_weight

    @staticmethod
    def edge_no_limit(trade):
        uvol_weight = max(0, math.sqrt(float(trade.uvol) - trade.hyper_parameter.minuvol))
        uoi_weight = max(0, math.sqrt(float(trade.uoi) - trade.hyper_parameter.minuoi))

        return uvol_weight + uoi_weight

    @staticmethod
    def tran_cost(shares, price):
        trade_fee = Decimal(0.005) * shares
        if price < 1:
            trade_fee *= price
        if trade_fee < 1:
            trade_fee = 1

        return abs(trade_fee)

    @staticmethod
    def slippage(shares):
        return Decimal(0.02) * shares

    def write_trade(self, position):
        if self.config.log['tradef'] is not None:
            with open(rf"{self.config.log['directory']}\{self.config.log['tradef']}", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                position.to_csv(writer)
