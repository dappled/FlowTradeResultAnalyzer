from decimal import Decimal


class Configuration(object):
    def __init__(self, *args, **kwargs):
        if len(kwargs) > 0:
            if len(args) != 0 or len(kwargs) > 1 or kwargs['config'] is None:
                raise Exception("incorrect input parameters")
            target_config = kwargs['config']
            self.name = target_config.name
            self.start_time = target_config.start_time
            self.end_time = target_config.end_time
            self.optimise_target = target_config.optimise_target
            self.detail_report = target_config.detail_report
            self.portfolio = target_config.portfolio
            self.log = target_config.log.copy()
        else:
            if len(args) != 14:
                raise Exception('incorrect input parameters: name, start_time, end_time, optimise_target, detail_report, directory, trade file, statistic file, equity graph, underwater graph, capital holding, mae, mte, mfe')
            i = 0
            self.name = args[i]
            i += 1
            self.start_time = args[i]
            i += 1
            self.end_time = args[i]
            i += 1
            self.optimise_target = args[i]
            i += 1
            self.detail_report = args[i]
            i += 1
            self.portfolio = PortfolioConfiguration()
            self.log = {"directory": args[i],
                        "tradef": args[i+1],
                        "statsf": args[i+2],
                        "equityg": args[i+3],
                        "underwaterg": args[i+4],
                        "capitalg": args[i+5],
                        "maeg": args[i+6],
                        "mteg": args[i+7],
                        "mfeg": args[i+8]}

    def turn_off_log(self):
        self.log["tradef"] = None
        self.log["statsf"] = None
        self.log["equityg"] = None
        self.log["underwaterg"] = None
        self.log["capitalg"] = None
        self.log["maeg"] = None
        self.log["mteg"] = None
        self.log["mfeg"] = None

class PortfolioConfiguration(object):
    def __init__(self):
        self.initial_capital = Decimal(1000000)
        self.leverage = Decimal(0.2)
        self.min_cash_hold = 0
        self.unit_cash_per_trade = Decimal(0.01) * self.initial_capital #10000
        self.max_cash_per_trade = Decimal(0.05) * self.initial_capital #50000
        self.min_cash_per_trade = max(100, Decimal(0.00) * self.initial_capital) #1000
        self.max_cash_per_name = Decimal(0.1) * self.initial_capital
        self.max_uvol_weight = float(self.max_cash_per_trade) / float(self.unit_cash_per_trade) / 2.0
        self.max_uoi_weight = self.max_uvol_weight
        self.max_trade_per_name = Decimal("inf") #3
        if self.max_cash_per_trade > self.max_cash_per_name:
            self.max_cash_per_name = self.max_cash_per_trade
        self.care_transaction_fee = True
        self.interest_rate = 0
        self.target_rate = 0
        self.borrow_rate_daily = Decimal((1 + 0.02) ** (1.0 / 252.0) - 1) # 0
        self.allow_borrow = Decimal("inf") #self.leverage * self.initial_capital
