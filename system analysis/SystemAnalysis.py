import pickle

from Launcher import *
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import multiprocessing
from matplotlib import cm
import random
import datetime
import statsmodels.stats.api as sms

def run_portfolio_simple(trade_info, parameter, config, print_result):
    config_local = Configuration.Configuration(config=config)
    if not print_result:
        config_local.turn_off_log()
    trades = signal_to_trades(trade_info, parameter)
    portfolio = Portfolio(trades, config_local)
    result = portfolio.run(config_local.start_time, config_local.end_time)
    return result


def run_portfolio2(rangevalue, trade_infos, raw_parameter, port_start_time, port_end_time, print_local, config):
    uvol_value, uoi_value = rangevalue
    parameter = raw_parameter.change_property("minuvol", uvol_value)
    parameter = parameter.change_property("minuoi", uoi_value)
    config_local = Configuration.Configuration(config=config)
    config_local.log['directory'] = rf'{config_local.log["directory"]}\{uvol_value},{uoi_value}'
    if not os.path.exists(config_local.log['directory']):
        os.makedirs(config_local.log['directory'])
    if not print_local:
        config_local.turn_off_log()
    trades = signal_to_trades(trade_infos, parameter)
    portfolio = Portfolio(trades, config_local)
    result = portfolio.run(port_start_time, port_end_time)
    # print(f"finished {parameter}")
    return uvol_value, uoi_value, result

def run_portfolio_vary_maxlossforce(maxlossforce, trade_infos, raw_parameter, port_start_time, port_end_time, print_local, config):
    parameter = raw_parameter.change_property("maxlossforce", -maxlossforce)
    config_local = Configuration.Configuration(config=config)
    config_local.log['directory'] = rf'{config_local.log["directory"]}\{round(maxlossforce, 2):.2f}'
    if not os.path.exists(config_local.log['directory']):
        os.makedirs(config_local.log['directory'])
    if not print_local:
        config_local.turn_off_log()
    trades = signal_to_trades(trade_infos, parameter)
    portfolio = Portfolio(trades, config_local)
    result = portfolio.run(port_start_time, port_end_time)
    # print(f"finished {parameter}")
    return maxlossforce, result

def run_portfolio_vary_trailingloss(maxloss, trade_infos, raw_parameter, port_start_time, port_end_time, print_local, config):
    parameter = raw_parameter.change_property("maxloss", -maxloss)
    config_local = Configuration.Configuration(config=config)
    config_local.log['directory'] = rf'{config_local.log["directory"]}\{round(maxloss, 2):.2f}'
    if not os.path.exists(config_local.log['directory']):
        os.makedirs(config_local.log['directory'])
    if not print_local:
        config_local.turn_off_log()
    trades = signal_to_trades(trade_infos, parameter)
    portfolio = Portfolio(trades, config_local)
    result = portfolio.run(port_start_time, port_end_time)
    # print(f"finished {parameter}")
    return maxloss, result

def monte_carlo_revenue(folder, daily_status, initial_capital, times, plot):
    run_time = max(times, math.factorial(len(daily_status)))
    all_potential_equity_lines = []
    all_potential_max_dd = []
    all_potential_avg_dd = []
    for i in range(1, run_time + 1):
        new_status = random.sample(daily_status, len(daily_status))
        max_drawdown, avg_drawdown, equity_line = drawdown_from_daily_status(new_status, initial_capital)
        all_potential_equity_lines.append([equity_line])
        all_potential_max_dd.append(max_drawdown)
        all_potential_avg_dd.append(avg_drawdown)

    max_drawdown_origin, avg_drawdown_origin = drawdown_from_daily_status(daily_status, initial_capital)
    max_dd_stats = sms.DescrStatsW(all_potential_max_dd)
    avg_dd_stats = sms.DescrStatsW(all_potential_avg_dd)
    sixty_confidence_max_dd = max_dd_stats.tconfint_mean(alpha=0.4, alternative="smaller")
    seventy_confidence_max_dd = max_dd_stats.tconfint_mean(alpha=0.3, alternative="smaller")
    eighty_confidence_max_dd = max_dd_stats.tconfint_mean(alpha=0.2, alternative="smaller")
    ninety_confidence_max_dd = max_dd_stats.tconfint_mean(alpha=0.1, alternative="smaller")
    ninetyfive_confidence_max_dd = max_dd_stats.tconfint_mean(alpha=0.05, alternative="smaller")
    ninetynine_confidence_max_dd = max_dd_stats.tconfint_mean(alpha=0.01, alternative="smaller")

    sixty_confidence_avg_dd = avg_dd_stats.tconfint_mean(alpha=0.4, alternative="smaller")
    seventy_confidence_avg_dd = avg_dd_stats.tconfint_mean(alpha=0.3, alternative="smaller")
    eighty_confidence_avg_dd = avg_dd_stats.tconfint_mean(alpha=0.2, alternative="smaller")
    ninety_confidence_avg_dd = avg_dd_stats.tconfint_mean(alpha=0.1, alternative="smaller")
    ninetyfive_confidence_avg_dd = avg_dd_stats.tconfint_mean(alpha=0.05, alternative="smaller")
    ninetynine_confidence_avg_dd = avg_dd_stats.tconfint_mean(alpha=0.01, alternative="smaller")

    # mean_max_dd = np.mean(all_potential_max_dd)
    # std_max_dd = np.std(all_potential_max_dd)
    # mean_avg_dd = np.mean(all_potential_avg_dd)
    # std_avg_dd = np.std(all_potential_avg_dd)
    # sixty_confidence_max_dd = mean_max_dd -

    report1 = pd.DataFrame.from_items([('Number of samples for Monte Carlo Analysis', [run_time]),
                                       ('Total Net Profit', [daily_status[-1][1] - initial_capital]),
                                       ('Final Account Equity', [daily_status[-1][1]])],
                                      orient='index', columns=["monte carlo"])
    report2 = pd.DataFrame.from_items([('original system', [max_drawdown_origin, avg_drawdown_origin]),
                                       ('Monte Carlo 60% confidence',
                                        [sixty_confidence_max_dd, sixty_confidence_avg_dd]),
                                       ('Monte Carlo 70% confidence',
                                        [seventy_confidence_max_dd, seventy_confidence_avg_dd]),
                                       ('Monte Carlo 80% confidence',
                                        [eighty_confidence_max_dd, eighty_confidence_avg_dd]),
                                       ('Monte Carlo 90% confidence',
                                        [ninety_confidence_max_dd, ninety_confidence_avg_dd]),
                                       ('Monte Carlo 95% confidence',
                                        [ninetyfive_confidence_max_dd, ninetyfive_confidence_avg_dd]),
                                       ('Monte Carlo 99% confidence',
                                        [ninetynine_confidence_max_dd, ninetynine_confidence_avg_dd])],
                                      orient='index', columns=['max drawdown', 'avg drawdown'])
    file_name = f'{folder}\Monte_Carlo.csv'
    report1.to_csv(file_name, header=False)
    report2.to_csv(file_name, mode='a')
    # plot
    if plot:
        11


def drawdown_from_daily_status(daily_status, initial_capital):
    default_status = [datetime.date(1970, 1, 1), 0, 0]
    max_drawdown = default_status
    max_drawdown_tmp = default_status
    sum_drawdown = 0
    drop_from_last_high = 0
    drop_from_last_high_tmp = 0
    previous_high_tmp = default_status
    status_count = 0
    status_len = len(daily_status)
    equity_line = []

    for status in daily_status:
        status_count += 1
        cash_value = status[1]

        # draw down
        if cash_value >= previous_high_tmp[1] or status_count == status_len:
            if drop_from_last_high_tmp <= drop_from_last_high:
                max_drawdown = max_drawdown_tmp
                drop_from_last_high = drop_from_last_high_tmp

            drop_from_last_high_tmp = 0
            previous_high_tmp = status
            max_drawdown_tmp = default_status
        else:
            diff = cash_value - previous_high_tmp[1]
            if diff < drop_from_last_high_tmp:
                drop_from_last_high_tmp = diff
                max_drawdown_tmp = status

            sum_drawdown += diff

        equity_line.append(status[1] - initial_capital)

    return max_drawdown, sum_drawdown / status_len, equity_line



if __name__ == "__main__":
    # setting
    test_to_run = 1
    optimise_target = 3  # 1 is total return, 2 is hit ratio, 3 is net return/max dd
    call_strategy = True
    chyper_parameter = HyperParameter(1, float("inf"), float("-inf"), float("-inf"), 40, 0, 0, 0, 1, float("inf"), 5)
    hyper_parameter = chyper_parameter if call_strategy else phyper_parameter
    print_result = True
    is_earning = False
    algo_days = 90
    start_time = pd.to_datetime("20130102", format='%Y%m%d')
    end_time = pd.to_datetime("20161202", format='%Y%m%d')
    source_directory = rf'C:\Temp\FlowTrade\TradeAlert\Report\backtest\testpython\SystemAnalysis'
    if not os.path.exists(source_directory):
        os.makedirs(source_directory)
    origin_config = Configuration.Configuration("13-16", start_time, end_time, optimise_target, True, source_directory,
                                                f"Trades.csv", f"Stats.csv", f"Equity.png", f'Underwater.png',
                                                f"Capital.png", f"MAE", f"MTE", f"MFE")
    earning_str = f"{'y' if is_earning else 'n'}"
    file = rf'C:\Temp\FlowTrade\TradeAlert\Report\backtest\MetaData13-16(E({earning_str})C(y)P(5)H(n))\{"BullCalls" if call_strategy else "BearPuts"}{algo_days}D(E({earning_str})C(y)P(5)H(' \
           rf'n))\TestingTrade{"BullCalls" if call_strategy else "BearPuts"}{algo_days}D(E({earning_str})C(y)P(5)H(n)).csv '
    trade_info = read_file(file)

    # # evaluate simple startegy no parameter optimisation, no trade filter, no exit
    # sub_directory_no_optimize = rf'{source_directory}\NoOptimize'
    # if not os.path.exists(sub_directory_no_optimize):
    #     os.makedirs(sub_directory_no_optimize)
    # config = Configuration.Configuration(config=origin_config)
    # config.log['directory'] = f'{sub_directory_no_optimize}'
    # run_portfolio_single(trade_info, hyper_parameter, config)
    #
    # # add transaction
    # config.portfolio.care_transaction_fee = True
    # sub_directory_no_optimize_transaction = rf'{source_directory}\AddCost'
    # if not os.path.exists(sub_directory_no_optimize_transaction):
    #     os.makedirs(sub_directory_no_optimize_transaction)
    # config.log['directory'] = f'{sub_directory_no_optimize_transaction}'
    # run_portfolio_single(trade_info, hyper_parameter, config)

    # # add parameter optimisation
    # sub_directory_optimize_parameter = rf'{source_directory}\OptimizeParameter'
    # if not os.path.exists(sub_directory_optimize_parameter):
    #     os.makedirs(sub_directory_optimize_parameter)
    # config = Configuration.Configuration(config=origin_config)
    # config.log['directory'] = rf'{sub_directory_optimize_parameter}'
    # uoi_range = np.concatenate([np.arange(0, 10, 0.5), np.arange(10, 16, 2)])  # np.arange(0, 15, 0.5)
    # uvol_range = np.concatenate([np.arange(0, 10, 0.5), np.arange(10, 16, 2)])
    # df_net_profit_divide_max_dd = pd.DataFrame(index=uoi_range, columns=uvol_range)
    # # df_net_profit_divide_max_dd = df_net_profit_divide_max_dd.fillna(0)
    # df_net_profit = pd.DataFrame(index=uoi_range, columns=uvol_range)
    # # df_net_profit = df_net_profit.fillna(0)
    # df_max_dd = pd.DataFrame(index=uoi_range, columns=uvol_range)
    # df_max_dd_to_capital = pd.DataFrame(index=uoi_range, columns=uvol_range)
    # # df_max_dd = df_max_dd.fillna(0)
    # df_avg_trade_net_profit = pd.DataFrame(index=uoi_range, columns=uvol_range)
    # # df_avg_trade_net_profit = df_avg_trade_net_profit.fillna(0)
    # df_total_trades = pd.DataFrame(index=uoi_range, columns=uvol_range)
    # df_total_trades = df_total_trades.fillna(0)
    # df_hitratio = pd.DataFrame(index=uoi_range, columns=uvol_range)
    # df_hitratio = df_hitratio.fillna(0)
    # # with futures.ProcessPoolExecutor(8) as pool:
    # with multiprocessing.Pool(8) as pool:
    #     for uvol_tmp, uoi_tmp, result in pool.imap_unordered(  # pool.map(
    #             functools.partial(run_portfolio2, trade_infos=trade_info, raw_parameter=hyper_parameter,
    #                               port_start_time=start_time, port_end_time=end_time, print_local=print_result,
    #                               config=config),
    #             product(uvol_range, uoi_range)):
    #         if result is None:
    #             print(1)
    #         profit = float(result.get_value('Total Net Profit', config.name))
    #         dd = float(result.get_value('Max Drawdown', config.name))
    #         dd_to_capital = float(result.get_value('Max Drawdown to capital used', config.name))
    #         hitratio = float(result.get_value('Percent Profitable', config.name))
    #         df_net_profit_divide_max_dd.set_value(uoi_tmp, uvol_tmp, profit / abs(dd))
    #         df_net_profit.set_value(uoi_tmp, uvol_tmp, profit)
    #         df_max_dd.set_value(uoi_tmp, uvol_tmp, dd)
    #         df_max_dd_to_capital.set_value(uoi_tmp, uvol_tmp, dd_to_capital)
    #         df_avg_trade_net_profit.set_value(uoi_tmp, uvol_tmp,
    #                                           float(result.get_value('Avg. Trade Net Profit', config.name)))
    #         df_total_trades.set_value(uoi_tmp, uvol_tmp, result.get_value('Total Number of Trades', config.name))
    #         df_hitratio.set_value(uoi_tmp, uvol_tmp, hitratio*100)
    #         print(f"finished {uvol_tmp},{uoi_tmp}: return({profit:.2f}),hit({hitratio*100:.2f}%)")
    #
    # uvol_range, uoi_range = np.meshgrid(uvol_range, uoi_range)
    #
    # matplotlib.rcParams['xtick.direction'] = 'out'
    # matplotlib.rcParams['ytick.direction'] = 'out'
    #
    # df_net_profit_divide_max_dd.to_csv(rf"{sub_directory_optimize_parameter}\net profit divided by max dd.csv")
    # fig_net_profit_divide_max_dd = plt.figure()
    # ax_net_profit_divide_max_dd = fig_net_profit_divide_max_dd.gca()
    # cs_net_profit_divide_max_dd = ax_net_profit_divide_max_dd.contour(uvol_range, uoi_range,
    #                                                                   df_net_profit_divide_max_dd)
    # ax_net_profit_divide_max_dd.clabel(cs_net_profit_divide_max_dd, inline=1, fontsize=10)
    # fig_net_profit_divide_max_dd.colorbar(cs_net_profit_divide_max_dd, shrink=0.8, extend='both')
    # ax_net_profit_divide_max_dd.set_title("net profit divide max dd")
    # ax_net_profit_divide_max_dd.set_xlabel("uvol")
    # ax_net_profit_divide_max_dd.set_ylabel("uoi")
    # fig_net_profit_divide_max_dd.savefig(rf'{sub_directory_optimize_parameter}\net profit divided by max dd.png')
    #
    # df_net_profit.to_csv(rf"{sub_directory_optimize_parameter}\net profit.csv")
    # fig_net_profit = plt.figure()
    # ax_net_profit = fig_net_profit.gca()
    # cs_net_profit = ax_net_profit.contour(uvol_range, uoi_range, df_net_profit)
    # ax_net_profit.clabel(cs_net_profit, inline=1, fontsize=10)
    # fig_net_profit.colorbar(cs_net_profit, shrink=0.8, extend='both')
    # ax_net_profit.set_title("net profit")
    # ax_net_profit.set_xlabel("uvol")
    # ax_net_profit.set_ylabel("uoi")
    # fig_net_profit.savefig(rf'{sub_directory_optimize_parameter}\net profit.png')
    #
    # df_max_dd.to_csv(rf"{sub_directory_optimize_parameter}\max dd.csv")
    # fig_max_dd = plt.figure()
    # ax_max_dd = fig_max_dd.gca()
    # cs_max_dd = ax_max_dd.contour(uvol_range, uoi_range, df_max_dd)
    # ax_max_dd.clabel(cs_max_dd, inline=1, fontsize=10)
    # fig_max_dd.colorbar(cs_max_dd, shrink=0.8, extend='both')
    # ax_max_dd.set_title("max dd")
    # ax_max_dd.set_xlabel("uvol")
    # ax_max_dd.set_ylabel("uoi")
    # fig_max_dd.savefig(rf'{sub_directory_optimize_parameter}\max dd.png')
    #
    # df_max_dd_to_capital.to_csv(rf"{sub_directory_optimize_parameter}\max dd to capital.csv")
    # fig_max_dd_to_capital = plt.figure()
    # ax_max_dd_to_capital = fig_max_dd_to_capital.gca()
    # cs_max_dd_to_capital = ax_max_dd_to_capital.contour(uvol_range, uoi_range, df_max_dd_to_capital)
    # ax_max_dd_to_capital.clabel(cs_max_dd_to_capital, inline=1, fontsize=10)
    # fig_max_dd_to_capital.colorbar(cs_max_dd_to_capital, shrink=0.8, extend='both')
    # ax_max_dd_to_capital.set_title("max dd to capital")
    # ax_max_dd_to_capital.set_xlabel("uvol")
    # ax_max_dd_to_capital.set_ylabel("uoi")
    # fig_max_dd_to_capital.savefig(rf'{sub_directory_optimize_parameter}\max dd to capital.png')
    #
    # df_avg_trade_net_profit.to_csv(rf"{sub_directory_optimize_parameter}\avg trade net profit.csv")
    # fig_avg_trade_net_profit = plt.figure()
    # ax_avg_trade_net_profit = fig_avg_trade_net_profit.gca()
    # cs_avg_trade_net_profit = ax_avg_trade_net_profit.contour(uvol_range, uoi_range, df_avg_trade_net_profit)
    # ax_avg_trade_net_profit.clabel(cs_avg_trade_net_profit, inline=1, fontsize=10)
    # fig_avg_trade_net_profit.colorbar(cs_avg_trade_net_profit, shrink=0.8, extend='both')
    # ax_avg_trade_net_profit.set_title("avg trade net profit")
    # ax_avg_trade_net_profit.set_xlabel("uvol")
    # ax_avg_trade_net_profit.set_ylabel("uoi")
    # fig_avg_trade_net_profit.savefig(rf'{sub_directory_optimize_parameter}\avg trade net profit.png')
    #
    # df_total_trades.to_csv(rf"{sub_directory_optimize_parameter}\total trades.csv")
    # fig_total_trades = plt.figure()
    # ax_total_trades = fig_total_trades.gca()
    # cs_total_trades = ax_total_trades.contour(uvol_range, uoi_range, df_total_trades)
    # ax_total_trades.clabel(cs_total_trades, inline=1, fontsize=10)
    # fig_total_trades.colorbar(cs_total_trades, shrink=0.8, extend='both')
    # ax_total_trades.set_title("total trades")
    # ax_total_trades.set_xlabel("uvol")
    # ax_total_trades.set_ylabel("uoi")
    # fig_total_trades.savefig(rf'{sub_directory_optimize_parameter}\total trades.png')
    #
    # df_hitratio.to_csv(rf"{sub_directory_optimize_parameter}\hit ratio.csv")
    # fig_hitratio = plt.figure()
    # ax_hitratio = fig_hitratio.gca()
    # cs_hitratio = ax_hitratio.contour(uvol_range, uoi_range, df_hitratio)
    # ax_hitratio.clabel(cs_hitratio, inline=1, fontsize=10)
    # fig_hitratio.colorbar(cs_hitratio, shrink=0.8, extend='both')
    # ax_hitratio.set_title("hit ratio")
    # ax_hitratio.set_xlabel("uvol")
    # ax_hitratio.set_ylabel("uoi")
    # fig_hitratio.savefig(rf'{sub_directory_optimize_parameter}\hit ratio.png')

    # add volatility filter

    # add stop loss
    # print_result = False
    # sub_directory_stop_loss = rf'{source_directory}\StopLossStep1'
    # if not os.path.exists(sub_directory_stop_loss):
    #     os.makedirs(sub_directory_stop_loss)
    # config = Configuration.Configuration(config=origin_config)
    # config.log['directory'] = f'{sub_directory_stop_loss}'
    # hyper_parameter = HyperParameter(1, float("inf"), float("-inf"), float("-inf"), 40, 8, 2, 0, 1, float("inf"), 5)
    # before_add_stop_loss = run_portfolio_simple(trade_info, hyper_parameter, config, print_result)

    # # stop loss step 2
    # print_result = False
    # sub_directory_stop_loss_step2 = rf'{source_directory}\StopLossStep2'
    # if not os.path.exists(sub_directory_stop_loss_step2):
    #     os.makedirs(sub_directory_stop_loss_step2)
    # config = Configuration.Configuration(config=origin_config)
    # config.log['directory'] = f'{sub_directory_stop_loss_step2}'
    # hyper_parameter = HyperParameter(1, float("inf"), float("-inf"), float("-inf"), 40, 8, 2, 0, 1, float("inf"), 5)
    # maxloss_range = np.concatenate([np.arange(0.01, 0.4, 0.01) , np.arange(0.4, 0.7, 0.1)])
    # dummy_range = np.arange(1,1)
    # df_net_profit_divide_max_dd = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    # df_net_profit = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    # df_max_dd = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    # df_max_dd_to_capital = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    # df_avg_trade_net_profit = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    # df_hitratio = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    # df_largest_loss = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    # df_largest_win = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    # with multiprocessing.Pool(8) as pool:
    #     for maxloss_tmp, result in pool.imap_unordered(  # pool.map(
    #             functools.partial(run_portfolio_vary_maxlossforce, trade_infos=trade_info, raw_parameter=hyper_parameter,
    #                               port_start_time=start_time, port_end_time=end_time, print_local=print_result,
    #                               config=config),
    #             maxloss_range):
    #         if result is None:
    #             print(1)
    #         profit = float(result.get_value('Total Net Profit', config.name))
    #         dd = float(result.get_value('Max Drawdown', config.name))
    #         dd_to_capital = float(result.get_value('Max Drawdown to capital used', config.name))
    #         hitratio = float(result.get_value('Percent Profitable', config.name))
    #         largest_loss_trade = float(result.get_value('Largest Losing Trade', config.name))
    #         largest_winning_trade = float(result.get_value('Largest Winning Trade', config.name))
    #         df_net_profit_divide_max_dd.set_value(1, maxloss_tmp, result.get_value('Net profit divided by dd', config.name))
    #         df_net_profit.set_value(1, maxloss_tmp, profit)
    #         df_max_dd.set_value(1, maxloss_tmp, dd)
    #         df_max_dd_to_capital.set_value(1, maxloss_tmp, dd_to_capital)
    #         df_avg_trade_net_profit.set_value(1, maxloss_tmp,
    #                                           float(result.get_value('Avg. Trade Net Profit', config.name)))
    #         df_hitratio.set_value(1, maxloss_tmp, hitratio*100)
    #         df_largest_win.set_value(1, maxloss_tmp, largest_winning_trade)
    #         df_largest_loss.set_value(1, maxloss_tmp, largest_loss_trade)
    #         print(f"finished {maxloss_tmp}: return({profit:.2f}),hit({hitratio*100:.2f}%)")
    #
    # maxloss_shape = maxloss_range.shape
    #
    # df_net_profit_divide_max_dd.to_csv(rf"{sub_directory_stop_loss_step2}\net profit divided by max dd.csv")
    # fig_net_profit_divide_max_dd = plt.figure()
    # ax_net_profit_divide_max_dd = fig_net_profit_divide_max_dd.gca()
    # ax_net_profit_divide_max_dd.plot(maxloss_range, np.array(df_net_profit_divide_max_dd).reshape(maxloss_shape))
    # ax_net_profit_divide_max_dd.axhline(before_add_stop_loss.get_value('Net profit divided by dd', config.name), c='r')
    # ax_net_profit_divide_max_dd.set_title("net profit divide max dd")
    # fig_net_profit_divide_max_dd.savefig(rf'{sub_directory_stop_loss_step2}\net profit divided by max dd.png')
    #
    # df_net_profit.to_csv(rf"{sub_directory_stop_loss_step2}\net profit.csv")
    # fig_net_profit = plt.figure()
    # ax_net_profit = fig_net_profit.gca()
    # ax_net_profit.plot(maxloss_range, np.array(df_net_profit).reshape(maxloss_shape))
    # ax_net_profit.axhline(before_add_stop_loss.get_value('Total Net Profit', config.name), c='r')
    # ax_net_profit.set_title("net profit")
    # fig_net_profit.savefig(rf'{sub_directory_stop_loss_step2}\net profit.png')
    #
    # df_max_dd.to_csv(rf"{sub_directory_stop_loss_step2}\max dd.csv")
    # fig_max_dd = plt.figure()
    # ax_max_dd = fig_max_dd.gca()
    # ax_max_dd.plot(maxloss_range, np.array(df_max_dd).reshape(maxloss_shape))
    # ax_max_dd.axhline(before_add_stop_loss.get_value('Max Drawdown', config.name), c='r')
    # ax_max_dd.set_title("max dd")
    # fig_max_dd.savefig(rf'{sub_directory_stop_loss_step2}\max dd.png')
    #
    # df_max_dd_to_capital.to_csv(rf"{sub_directory_stop_loss_step2}\max dd to capital.csv")
    # fig_max_dd_to_capital = plt.figure()
    # ax_max_dd_to_capital = fig_max_dd_to_capital.gca()
    # ax_max_dd_to_capital.plot(maxloss_range, np.array(df_max_dd_to_capital).reshape(maxloss_shape))
    # ax_max_dd_to_capital.axhline(before_add_stop_loss.get_value('Max Drawdown to capital used', config.name), c='r')
    # ax_max_dd_to_capital.set_title("max dd to capital")
    # fig_max_dd_to_capital.savefig(rf'{sub_directory_stop_loss_step2}\max dd to capital.png')
    #
    # df_avg_trade_net_profit.to_csv(rf"{sub_directory_stop_loss_step2}\avg trade net profit.csv")
    # fig_avg_trade_net_profit = plt.figure()
    # ax_avg_trade_net_profit = fig_avg_trade_net_profit.gca()
    # ax_avg_trade_net_profit.plot(maxloss_range, np.array(df_avg_trade_net_profit).reshape(maxloss_shape))
    # ax_avg_trade_net_profit.axhline(before_add_stop_loss.get_value('Avg. Trade Net Profit', config.name), c='r')
    # ax_avg_trade_net_profit.set_title("avg trade net profit")
    # fig_avg_trade_net_profit.savefig(rf'{sub_directory_stop_loss_step2}\avg trade net profit.png')
    #
    # df_hitratio.to_csv(rf"{sub_directory_stop_loss_step2}\hit ratio.csv")
    # fig_hitratio = plt.figure()
    # ax_hitratio = fig_hitratio.gca()
    # ax_hitratio.plot(maxloss_range, np.array(df_hitratio).reshape(maxloss_shape))
    # ax_hitratio.axhline(float(before_add_stop_loss.get_value('Percent Profitable', config.name)) * 100, c='r')
    # ax_hitratio.set_title("hit ratio")
    # fig_hitratio.savefig(rf'{sub_directory_stop_loss_step2}\hit ratio.png')
    #
    # df_largest_win.to_csv(rf"{sub_directory_stop_loss_step2}\max win.csv")
    # fig_largest_win = plt.figure()
    # ax_largest_win = fig_largest_win.gca()
    # ax_largest_win.plot(maxloss_range, np.array(df_largest_win).reshape(maxloss_shape))
    # ax_largest_win.axhline(before_add_stop_loss.get_value('Largest Winning Trade', config.name), c='r')
    # ax_largest_win.set_title("max win")
    # fig_largest_win.savefig(rf'{sub_directory_stop_loss_step2}\max win.png')
    #
    # df_largest_loss.to_csv(rf"{sub_directory_stop_loss_step2}\max loss.csv")
    # fig_largest_loss = plt.figure()
    # ax_largest_loss = fig_largest_loss.gca()
    # ax_largest_loss.plot(maxloss_range, np.array(df_largest_loss).reshape(maxloss_shape))
    # ax_largest_loss.axhline(before_add_stop_loss.get_value('Largest Losing Trade', config.name), c='r')
    # ax_largest_loss.set_title("max loss")
    # fig_largest_loss.savefig(rf'{sub_directory_stop_loss_step2}\max loss.png')

    # compare
    print_result = True
    sub_directory_stop_loss_cmp = rf'{source_directory}\StopLossStepCmp'
    if not os.path.exists(sub_directory_stop_loss_cmp):
        os.makedirs(sub_directory_stop_loss_cmp)
    config = Configuration.Configuration("add stop loss", start_time, end_time, optimise_target, True,
                                        sub_directory_stop_loss_cmp,
                                        f"TradesStopLoss.csv", f"StatsStopLoss.csv", f"EquityStopLoss.png", f'UnderwaterStopLoss.png',
                                        f"CapitalStopLoss.png", f"MAEStopLoss", f"MTEStopLoss", f"MFEStopLoss")
    hyper_parameter = HyperParameter(1, float("inf"), -0.15, float("-inf"), 40, 8, 2, 0, 1, float("inf"), 5)
    after_add_simple_stop_loss_config = config
    after_add_simple_stop_loss = run_portfolio_simple(trade_info, hyper_parameter, config, print_result)
    # compare_table = pd.concat([before_add_stop_loss, after_add_simple_stop_loss], axis=1)
    # compare_table.to_csv(rf"{sub_directory_stop_loss_cmp}\compare add stop loss.csv")

    # trailing loss
    sub_directory_trailing_loss = rf'{source_directory}\TrailingLoss'
    if not os.path.exists(sub_directory_trailing_loss):
        os.makedirs(sub_directory_trailing_loss)
    config = Configuration.Configuration(config=origin_config)
    config.log['directory'] = f'{sub_directory_trailing_loss}'
    hyper_parameter = HyperParameter(1, float("inf"), -0.15, float("-inf"), 40, 8, 2, 0, 1, float("inf"), 5)
    maxloss_range = np.concatenate([np.arange(0.01, 0.4, 0.01) , np.arange(0.4, 0.7, 0.1)])
    dummy_range = np.arange(1,1)
    df_net_profit_divide_max_dd = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    df_net_profit = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    df_max_dd = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    df_max_dd_to_capital = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    df_avg_trade_net_profit = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    df_hitratio = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    df_largest_loss = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    df_largest_win = pd.DataFrame(index=dummy_range, columns=maxloss_range)
    with multiprocessing.Pool(8) as pool:
        for maxloss_tmp, result in pool.imap_unordered(  # pool.map(
                functools.partial(run_portfolio_vary_trailingloss, trade_infos=trade_info, raw_parameter=hyper_parameter,
                                  port_start_time=start_time, port_end_time=end_time, print_local=print_result,
                                  config=config),
                maxloss_range):
            if result is None:
                print(1)
            profit = float(result.get_value('Total Net Profit', config.name))
            dd = float(result.get_value('Max Drawdown', config.name))
            dd_to_capital = float(result.get_value('Max Drawdown to capital used', config.name))
            hitratio = float(result.get_value('Percent Profitable', config.name))
            largest_loss_trade = float(result.get_value('Largest Losing Trade', config.name))
            largest_winning_trade = float(result.get_value('Largest Winning Trade', config.name))
            df_net_profit_divide_max_dd.set_value(1, maxloss_tmp, result.get_value('Net profit divided by dd', config.name))
            df_net_profit.set_value(1, maxloss_tmp, profit)
            df_max_dd.set_value(1, maxloss_tmp, dd)
            df_max_dd_to_capital.set_value(1, maxloss_tmp, dd_to_capital)
            df_avg_trade_net_profit.set_value(1, maxloss_tmp,
                                              float(result.get_value('Avg. Trade Net Profit', config.name)))
            df_hitratio.set_value(1, maxloss_tmp, hitratio*100)
            df_largest_win.set_value(1, maxloss_tmp, largest_winning_trade)
            df_largest_loss.set_value(1, maxloss_tmp, largest_loss_trade)
            print(f"finished {maxloss_tmp}: return({profit:.2f}),hit({hitratio*100:.2f}%)")

    maxloss_shape = maxloss_range.shape

    df_net_profit_divide_max_dd.to_csv(rf"{sub_directory_trailing_loss}\net profit divided by max dd.csv")
    fig_net_profit_divide_max_dd = plt.figure()
    ax_net_profit_divide_max_dd = fig_net_profit_divide_max_dd.gca()
    ax_net_profit_divide_max_dd.plot(maxloss_range, np.array(df_net_profit_divide_max_dd).reshape(maxloss_shape))
    ax_net_profit_divide_max_dd.axhline(after_add_simple_stop_loss.get_value('Net profit divided by dd', after_add_simple_stop_loss_config.name), c='r')
    ax_net_profit_divide_max_dd.set_title("net profit divide max dd")
    fig_net_profit_divide_max_dd.savefig(rf'{sub_directory_trailing_loss}\net profit divided by max dd.png')

    df_net_profit.to_csv(rf"{sub_directory_trailing_loss}\net profit.csv")
    fig_net_profit = plt.figure()
    ax_net_profit = fig_net_profit.gca()
    ax_net_profit.plot(maxloss_range, np.array(df_net_profit).reshape(maxloss_shape))
    ax_net_profit.axhline(after_add_simple_stop_loss.get_value('Total Net Profit', after_add_simple_stop_loss_config.name), c='r')
    ax_net_profit.set_title("net profit")
    fig_net_profit.savefig(rf'{sub_directory_trailing_loss}\net profit.png')

    df_max_dd.to_csv(rf"{sub_directory_trailing_loss}\max dd.csv")
    fig_max_dd = plt.figure()
    ax_max_dd = fig_max_dd.gca()
    ax_max_dd.plot(maxloss_range, np.array(df_max_dd).reshape(maxloss_shape))
    ax_max_dd.axhline(after_add_simple_stop_loss.get_value('Max Drawdown', after_add_simple_stop_loss_config.name), c='r')
    ax_max_dd.set_title("max dd")
    fig_max_dd.savefig(rf'{sub_directory_trailing_loss}\max dd.png')

    df_max_dd_to_capital.to_csv(rf"{sub_directory_trailing_loss}\max dd to capital.csv")
    fig_max_dd_to_capital = plt.figure()
    ax_max_dd_to_capital = fig_max_dd_to_capital.gca()
    ax_max_dd_to_capital.plot(maxloss_range, np.array(df_max_dd_to_capital).reshape(maxloss_shape))
    ax_max_dd_to_capital.axhline(after_add_simple_stop_loss.get_value('Max Drawdown to capital used', after_add_simple_stop_loss_config.name), c='r')
    ax_max_dd_to_capital.set_title("max dd to capital")
    fig_max_dd_to_capital.savefig(rf'{sub_directory_trailing_loss}\max dd to capital.png')

    df_avg_trade_net_profit.to_csv(rf"{sub_directory_trailing_loss}\avg trade net profit.csv")
    fig_avg_trade_net_profit = plt.figure()
    ax_avg_trade_net_profit = fig_avg_trade_net_profit.gca()
    ax_avg_trade_net_profit.plot(maxloss_range, np.array(df_avg_trade_net_profit).reshape(maxloss_shape))
    ax_avg_trade_net_profit.axhline(after_add_simple_stop_loss.get_value('Avg. Trade Net Profit', after_add_simple_stop_loss_config.name), c='r')
    ax_avg_trade_net_profit.set_title("avg trade net profit")
    fig_avg_trade_net_profit.savefig(rf'{sub_directory_trailing_loss}\avg trade net profit.png')

    df_hitratio.to_csv(rf"{sub_directory_trailing_loss}\hit ratio.csv")
    fig_hitratio = plt.figure()
    ax_hitratio = fig_hitratio.gca()
    ax_hitratio.plot(maxloss_range, np.array(df_hitratio).reshape(maxloss_shape))
    ax_hitratio.axhline(float(after_add_simple_stop_loss.get_value('Percent Profitable', after_add_simple_stop_loss_config.name)) * 100, c='r')
    ax_hitratio.set_title("hit ratio")
    fig_hitratio.savefig(rf'{sub_directory_trailing_loss}\hit ratio.png')

    df_largest_win.to_csv(rf"{sub_directory_trailing_loss}\max win.csv")
    fig_largest_win = plt.figure()
    ax_largest_win = fig_largest_win.gca()
    ax_largest_win.plot(maxloss_range, np.array(df_largest_win).reshape(maxloss_shape))
    ax_largest_win.axhline(after_add_simple_stop_loss.get_value('Largest Winning Trade', after_add_simple_stop_loss_config.name), c='r')
    ax_largest_win.set_title("max win")
    fig_largest_win.savefig(rf'{sub_directory_trailing_loss}\max win.png')

    df_largest_loss.to_csv(rf"{sub_directory_trailing_loss}\max loss.csv")
    fig_largest_loss = plt.figure()
    ax_largest_loss = fig_largest_loss.gca()
    ax_largest_loss.plot(maxloss_range, np.array(df_largest_loss).reshape(maxloss_shape))
    ax_largest_loss.axhline(after_add_simple_stop_loss.get_value('Largest Losing Trade', after_add_simple_stop_loss_config.name), c='r')
    ax_largest_loss.set_title("max loss")
    fig_largest_loss.savefig(rf'{sub_directory_trailing_loss}\max loss.png')

    # plt.show()
    print("finished")
    input()
