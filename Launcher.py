import csv
import functools
import os
from builtins import float, enumerate
from collections import namedtuple
from concurrent import futures
from time import time

from mpl_toolkits.mplot3d import axes3d, Axes3D
import hyperopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperopt import STATUS_FAIL
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp, fmin

import Configuration
from HyperParameter import HyperParameter
from Portfolio import Portfolio
from RawData import RawData
from Utils import *
import hyperopt.mongoexp
from itertools import product

trade = namedtuple('Trade', ['signal', 'hyper_parameter', 'prices'])


def read_file(file_name):
    i = 0
    raw_signals = []
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if not line:
                break
            if i > 1:
                # if line[-1]:
                raw_signals.append(RawData(line[0:15] + [line[26]], line[28:]))
            i += 1
    return raw_signals


def accept_signal(trade_signal, parameter):
    iv = perc_to_float(trade_signal.iv)
    if iv < parameter.volmin:
        return False
    elif iv > parameter.volmax:
        return False
    elif float(trade_signal.uvol) < parameter.minuvol:
        return False
    elif float(trade_signal.uoi) < parameter.minuoi:
        return False
    elif float(trade_signal.origin_price) < parameter.pricemin:
        return False
    else:
        return True


def signal_to_trades(trade_infos, parameter):
    trades = []
    for raw_data in trade_infos:
        # if raw_data.name == "XLFS":
        #     print(1)
        if accept_signal(raw_data, parameter):
            raw_data.hyper_parameter = parameter
            trades.append(raw_data)
    return trades


def signal_param_table():
    output_file = fr"{sub_directory}\{filew2}{algo_name}.csv"
    config_local = Configuration.Configuration(config=config)
    if print_result:
        config_local.log['output file'] = output_file
        try:
            os.remove(output_file)
        except OSError:
            pass
    else:
        output_file = None
    returns = {v: {} for v in param_range}
    with futures.ProcessPoolExecutor(8) as pool:
        for i, value, result in pool.map(
                functools.partial(run_portfolio, trade_infos=trade_info, raw_parameter=hyper_parameter, name_to_change=param_name[0], port_start_time=start_time, port_end_time=end_time,
                                  print_result_local=print_result, output_file=filew, config = config),
                enumerate(param_range)):
            if returns[value] != {}:
                raise Exception("should not be here")
            returns[value] = result[0]
            if output_file is not None:
                with open(output_file, 'a', newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    if i == 0:
                        writer.writerow((param_name, "total_return", "cash_return", "borrow_cost", "tran_cost", "trades"))
                    writer.writerow((value,) + result)
            print(f"{value:.2f}:{result[0]}")

    # plot
    plt.plot(list(returns.keys()), list(returns.values()))
    if print_result:
        plot_file = fr"{sub_directory}\{algo_name}.png"
        plt.savefig(plot_file)
    else:
        plt.show()


def run_portfolio(rangevalue, trade_infos, raw_parameter, name_to_change, port_start_time, port_end_time, print_result_local, output_file, config):
    i, value = rangevalue
    parameter = raw_parameter.change_property(name_to_change, value)
    config_local = Configuration.Configuration(config=config)
    if print_result_local:
        config_local.log['tradef'] = fr"{sub_directory}\{output_file}({parameter}).csv"
        config_local.log['capitalg'] = fr"{sub_directory}\{output_file}({parameter})Capital.png"
    else:
        config_local.log['tradef'] = None
        config_local.log['capitalg'] = None
    trades = signal_to_trades(trade_infos, parameter)
    portfolio = Portfolio(trades, config_local)
    result = portfolio.run(port_start_time, port_end_time)
    # print(f"finished {parameter}")
    return i, value, result

print_times = 0

def run_portfolio_single(trade_info, parameter, config, print_result):
    global print_times
    config_local = Configuration.Configuration(config=config)
    if not print_result:
        config_local.turn_off_log()
    trades = signal_to_trades(trade_info, parameter)
    portfolio = Portfolio(trades, config_local)
    result = portfolio.run(config_local.start_time, config_local.end_time)
    print_times += 1
    if config_local.detail_report:
        if result is None:
            print("error")
        net_profit = result.get_value('Total Net Profit', config_local.name)
        max_dd = result.get_value('Max Drawdown', config_local.name)
        hratio = result.get_value('Percent Profitable', config_local.name)
        print(f"{print_times}: finished {parameter}: return({net_profit:.2f}),hratio({hratio*100:.2f}%),maxdd({max_dd:.2f})")
        if config_local.optimise_target == 1:
            result = net_profit
        elif config_local.optimise_target == 2:
            result = hratio
        elif config_local.optimise_target == 3:
            result = net_profit / abs(max_dd)
        else:
            raise Exception("should not be here")
    else:
        print(f"{print_times}:finished {parameter}: return({result:.2f})")
    return result


def run_portfolio_wrapper_hyperopt(params):
    parameter = hyper_parameter
    for p in param_name:
        parameter = parameter.change_property(p, params[p])
    result = run_portfolio_single(trade_info, parameter, config)
    if result == Decimal('-inf'):
        return {'loss': Decimal('-inf'), 'status': STATUS_FAIL}
    return {'loss': -result, 'status': STATUS_OK}


def hyperopt_test():
    best_return = Decimal('-inf')
    best_parameter = None
    output_file_summary = fr"{sub_directory}\{algo_name}summary.csv"
    with open(output_file_summary, 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("unit_trade", "max_trade", param_name, "statics"))

    for tc in trade_capital_combo:
        global unit_trade_capital
        global max_trade_capital
        unit_trade_capital = tc[0]
        max_trade_capital = tc[1]

        p_name = f'{unit_trade_capital},{max_trade_capital}'
        sub_directory_new = fr'{sub_directory}\{p_name}'
        if not os.path.exists(sub_directory_new):
            os.makedirs(sub_directory_new)

        trials = hyperopt.Trials()
        best = fmin(run_portfolio_wrapper_hyperopt, space, algo_use, max_run_times, trials)
        best = hyperopt_best_to_real_best(best)

        print(f'{p_name}:best:{best}')

        cols = len(param_name)
        f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))
        for idx, val in enumerate(param_name):
            xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
            ys = [-t['result']['loss'] for t in trials.trials]
            xs, ys = zip(*sorted(zip(xs, ys)))
            axes[idx].scatter(xs, ys, color='k', s=20)
            axes[idx].set_title(val)
        plot_file = fr"{sub_directory_new}\({p_name}){algo_name}.png"
        plt.savefig(plot_file)

        # xs = np.array([t['misc']['vals'][param_name] for t in trials.trials]).ravel()
        # ys = [-t['result']['loss'] for t in trials.trials]
        # xs, ys = zip(*sorted((zip(xs, ys))))
        # plt.plot(xs, ys)

        xs = []
        ys = []
        for trial in trials.trials:
            parameter = {k: v[0] for k, v in trial['misc']['vals'].items()}
            parameter = hyperopt_best_to_real_best(parameter)
            xs.append(tuple(parameter[x] for x in param_name))
            ys.append(-trial['result']['loss'])
        ys, xs = zip(*sorted(zip(ys, xs), reverse=True))
        if ys[0] > best_return:
            best_parameter = xs[0]
            best_return = ys[0]
        output_file = fr"{sub_directory_new}\{filew2}({p_name}).csv"
        with open(output_file, 'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(tuple(param_name)+("result",))
            for i in range(0, len(xs)):
                writer.writerow(xs[i]+(ys[i],))

        with open(output_file_summary, 'a', newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow((unit_trade_capital, max_trade_capital, xs[0], ys[0]))

    print(f"best parameter:{best_parameter}, best result: {best_return}")


def hyperopt_best_to_real_best(best):
    for pn in param_name_need_transform:
        best[pn] = param_name_need_transform_ranges[pn][best[pn]]
    return best


if __name__ == "__main__":
    trade_capital_combo = ((10000, 50000),) # ((5000,25000),(10000, 50000), (20000, 100000),(40000, 200000))
    unit_trade_capital = 10000
    max_trade_capital = 50000
    param_name_need_transform_ranges = {'std': np.arange(0.5, 2.5 + 0.1, 0.1), 'maxloss': np.arange(-0.05, -0.15 - 0.01, -0.01), 'maxholding': np.arange(1, 20 + 1, 1), 'volmax': [50,60,70,80,90,100,150,100000]}
    param_name_need_transform = ['std', 'maxloss', 'maxholding', 'volmax']
    space = {
        'std': hp.choice('std', param_name_need_transform_ranges['std']),
        'maxloss': hp.choice('maxloss', param_name_need_transform_ranges['maxloss']),
        'maxholding': hp.choice('maxholding', param_name_need_transform_ranges['maxholding']),
        'stdmul': hp.choice('stdmul', [0, 1, 2, 3]),
        'minuvol': hp.uniform('minuvol', 0, 20),
        'minuoi': hp.uniform('minuoi', 0, 20),
        # 'volmin': hp.uniform('volmin', 1, 20),
        # 'volmax': hp.uniform('volmax', 50, 170)
        'volmax': hp.choice('volmax', param_name_need_transform_ranges['volmax'])
    }
    test_to_run = 1 # 1 is single, 2 is table to 1 parameter, 3 is all parameter optimise
    algo_to_run = 1 # 1 is tpe, 2 is rand
    optimise_target = 3 # 1 is total return, 2 is hit ratio, 3 is net return/max dd
    call_strategy = True
    print_result = True
    is_earning = False
    algo_days = 90
    max_run_times = 1000
    start_time = pd.to_datetime("20130102", format='%Y%m%d')
    end_time = pd.to_datetime("20161202", format='%Y%m%d')
    source_directory = rf'C:\Temp\FlowTrade\TradeAlert\Report\backtest\testpython'
    if not os.path.exists(source_directory):
        os.makedirs(source_directory)
    config_name = "15-16EntryOn13-16"
    config = Configuration.Configuration(config_name, start_time, end_time, optimise_target, True, source_directory, f"Trades({config_name}).csv", f"Stats({config_name}).csv", f"Equity({config_name}).png", f'Underwater({config_name}).png', f"Capital({config_name}).png")
    start_time_pool = time()
    param_name = list(space.keys())

    rangestart = 0.5
    rangeend = 2.5
    stepsize = 0.1
    earning_str = f"{'y' if is_earning else 'n'}"
    algo_use = hyperopt.rand.suggest if algo_to_run == 2 else hyperopt.tpe.suggest  # hyperopt.rand/tpe

    chyper_parameter = HyperParameter(1, 1.7,-0.09,5,6,2,3,1,100, 5)
    phyper_parameter = HyperParameter(-1, 0.6,-0.09,3,7,8,0,1,90, 5)

    param_range = np.arange(rangestart, rangeend + stepsize, stepsize)
    range2 = enumerate(param_range)
    algo_name = f"{'Call' if call_strategy else 'Put'}E({earning_str}){algo_days}D{unit_trade_capital/1000 if test_to_run == 1 else ''}{'K' if test_to_run == 1 else ''}"  # 'NoUpperVolLimit' f"{param_name}({rangestart}-{rangeend})"
    file = rf'C:\Temp\FlowTrade\TradeAlert\Report\backtest\MetaData13-16(E({earning_str})C(y)P(5)H(n))\{"BullCalls" if call_strategy else "BearPuts"}{algo_days}D(E({earning_str})C(y)P(5)H(' \
           rf'n))\TestingTrade{"BullCalls" if call_strategy else "BearPuts"}{algo_days}D(E({earning_str})C(y)P(5)H(n)).csv '
    sub_directory = fr'{source_directory}\{algo_name}{"RetToDD" if optimise_target == 3 else "HitRatio" if optimise_target == 2 else "Return"}{"" if test_to_run != 3 else "Rnd" if algo_to_run == 2 else "Tpe"}{config_name}'
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
    filew = algo_name
    filew2 = f'{algo_name}Results'
    hyper_parameter = chyper_parameter if call_strategy else phyper_parameter

    trade_info = read_file(file)
    if test_to_run == 1:
        config.log["directory"] = sub_directory
        print(str(run_portfolio_single(trade_info, hyper_parameter, config)))
    elif test_to_run == 2:
        signal_param_table()
    elif test_to_run == 3:
        hyperopt_test()
    else:
        raise Exception(f"unkown test_to_run {test_to_run}")
    print(f"took {time() - start_time_pool:.2f} secs")
