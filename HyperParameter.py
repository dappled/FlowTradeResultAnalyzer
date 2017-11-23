class HyperParameter(object):
    __slot__ = ['direction', 'std', 'maxlossforce', 'maxloss', 'maxholding', 'minuvol', 'minuoi', 'stdmul', 'volmin', 'volmax', 'pricemin']

    def __init__(self, direction, std, maxlossforce, maxloss, maxholding, minuvol, minuoi, stdmul, volmin, volmax, pricemin):
        self.direction = direction
        self.std = std
        self.maxlossforce = maxlossforce
        self.maxloss = maxloss
        self.maxholding = maxholding
        self.minuvol = minuvol
        self.minuoi = minuoi
        self.stdmul = stdmul
        self.volmin = volmin
        self.volmax = volmax
        self.pricemin = pricemin

    def change_property(self, key, value):
        if key == "std":
            return HyperParameter(self.direction, value, self.maxlossforce, self.maxloss, self.maxholding, self.minuvol, self.minuoi, self.stdmul, self.volmin, self.volmax, self.pricemin)
        elif key == "maxlossforce":
            return HyperParameter(self.direction, self.std, value, self.maxloss, self.maxholding, self.minuvol, self.minuoi, self.stdmul, self.volmin, self.volmax, self.pricemin)
        elif key == "maxloss":
            return HyperParameter(self.direction, self.std, self.maxlossforce, value, self.maxholding, self.minuvol, self.minuoi, self.stdmul, self.volmin, self.volmax, self.pricemin)
        elif key == "maxholding":
            return HyperParameter(self.direction, self.std, self.maxlossforce, self.maxloss, value, self.minuvol, self.minuoi, self.stdmul, self.volmin, self.volmax, self.pricemin)
        elif key == "minuvol":
            return HyperParameter(self.direction, self.std, self.maxlossforce, self.maxloss, self.maxholding, value, self.minuoi, self.stdmul, self.volmin, self.volmax, self.pricemin)
        elif key == "minuoi":
            return HyperParameter(self.direction, self.std, self.maxlossforce, self.maxloss, self.maxholding, self.minuvol, value, self.stdmul, self.volmin, self.volmax, self.pricemin)
        elif key == "stdmul":
            return HyperParameter(self.direction, self.std, self.maxlossforce, self.maxloss, self.maxholding, self.minuvol, self.minuoi, value, self.volmin, self.volmax, self.pricemin)
        elif key == "volmin":
            return HyperParameter(self.direction, self.std, self.maxlossforce, self.maxloss, self.maxholding, self.minuvol, self.minuoi, self.stdmul, value, self.volmax, self.pricemin)
        elif key == "volmax":
            return HyperParameter(self.direction, self.std, self.maxlossforce, self.maxloss, self.maxholding, self.minuvol, self.minuoi, self.stdmul, self.volmin, value, self.pricemin)
        elif key == "pricemin":
            return HyperParameter(self.direction, self.std, self.maxlossforce, self.maxloss, self.maxholding, self.minuvol, self.minuoi, self.stdmul, self.volmin, self.volmax, value)
        else:
            raise Exception("should not be here")

    def __repr__(self, **kwargs):
        return str(self)

    def __str__(self, **kwargs):
        return f"{self.direction},{self.std:.2f},{self.maxlossforce:.2f},{self.maxloss:.2f},{self.maxholding},{self.minuvol:.2f},{self.minuoi:.2f},{self.stdmul},{self.volmin:.2f},{self.volmax:.2f},{self.pricemin}"
