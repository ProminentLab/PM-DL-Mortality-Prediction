import numpy as np

class LinearDecay:
    def __init__(self, alpha=0, beta=0):
        self.alpha = alpha
        self.beta = beta
        self.setMeta()

    def __str__(self):
        return str(self.meta)

    def __repr__(self):
        return str(self.meta)

    def loadFromDict(self, dictionary):
        self.alpha = dictionary['alpha']
        self.beta = dictionary['beta']
        self.setMeta()

    def setMeta(self):
        self.meta = dict()
        self.meta["DecayFunction"] = "LinearDecay"
        self.meta["alpha"] = self.alpha
        self.meta["beta"] = self.beta

    def decay(self, t):
        val = self.beta - (t * self.alpha)
        if val > 0:
            return val
        else:
            return 0.0

    def toJSON(self):
        return self.meta

class ExponentialDecay:
    def __init__(self, max=1.0, beta=1.0, thresh=0.01):
        self.beta = beta
        self.alpha = self.estimate(max,thresh)
        self.setMeta()

    def estimate(self, max, thresh):
        x2 = np.log(self.beta - thresh)
        x2 = np.divide(x2, max)
        return np.power(10,x2)

    def __str__(self):
        return str(self.meta)

    def __repr__(self):
        return str(self.meta)

    def loadFromDict(self, dictionary):
        self.alpha = dictionary['alpha']
        self.beta = dictionary['beta']
        self.setMeta()

    def setMeta(self):
        self.meta = dict()
        self.meta["DecayFunction"] = "ExponentialDecay"
        self.meta["alpha"] = self.alpha
        self.meta["beta"] = self.beta

    def decay(self, t):
        """  y=b(1-a)^t """
        val = self.beta * np.power((1-self.alpha), t)
        if (val > 0) and not np.isinf(val):
            return val
        else:
            return 0.0

    def toJSON(self):
        return self.meta



class LogDecay:
    """ for  0 < b < 1 , the function decays as  x  increases. (E.g., log1/2(1) > log1/2(2) > log1/2(3) .) Smaller values of  b  lead to slower rates of decay. """
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.setMeta()

    def __str__(self):
        return str(self.meta)

    def __repr__(self):
        return str(self.meta)

    def loadFromDict(self, dictionary):
        self.alpha = dictionary['alpha']
        self.setMeta()

    def setMeta(self):
        self.meta = dict()
        self.meta["DecayFunction"] = "LogDecay"
        self.meta["alpha"] = self.alpha

    def decay(self, t):
        val = np.log(t)/np.log(self.alpha)
        if not np.isinf(val) and not np.isnan(val):
            return  val
        else:
            return 0.0

    def toJSON(self):
        return self.meta

REGISTER = {
    'LinearDecay' : LinearDecay,
    'ExponentialDecay' : ExponentialDecay,
    'LogDecay' : LogDecay
}