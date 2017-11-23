class RawData:
    __slot__ = ['date', 'name', 'vol', 'adv', 'oi', 'uvol', 'uoi', 'vd', 'iv', 'ivchg', 'edate', 'conf', 'rate', 'vola', 'cap', 'origin_price', 'prices', 'hyper_parameter']

    def __init__(self, singal, prices):
        i = 0
        self.date = singal[i]
        i += 1
        self.name = singal[i]
        i += 1
        self.vol = singal[i]
        i += 1
        self.adv = singal[i]
        i += 1
        self.oi = singal[i]
        i += 1
        self.uvol = singal[i]
        i += 1
        self.uoi = singal[i]
        i += 1
        self.vd = singal[i]
        i += 1
        self.iv = singal[i]
        i += 1
        self.ivchg = singal[i]
        i += 1
        self.edate = singal[i]
        i += 1
        self.conf = singal[i]
        i += 1
        self.rate = singal[i]
        i += 1
        self.vola = singal[i]
        i += 1
        self.cap = singal[i]
        i += 1
        self.origin_price = singal[i]
        i += 1
        self.prices = prices
        self.hyper_parameter = None

    def have_valid_prices(self):
        p = self.prices
        return all(i for i in p)

    def signal_to_tuple(self):
        return self.date, self.name, self.vol, self.adv, self.oi, self.uvol, self.uoi, self.vd, self.iv, self.ivchg, self.edate, self.conf, self.rate, self.vola, self.cap, self.origin_price
