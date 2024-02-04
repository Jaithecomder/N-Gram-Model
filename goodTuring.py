from sklearn.linear_model import LinearRegression
import numpy as np

class GoodTuring:
    def __init__(self, n, nGramsDict):
        self.n = n
        self.nGramsDict = nGramsDict

    def getNr(self):
        freqDict = {}
        for nGram in self.nGramsDict:
            freq = self.nGramsDict[nGram]
            if freq in freqDict:
                freqDict[freq] += 1
            else:
                freqDict[freq] = 1
        return dict(sorted(freqDict.items()))
    
    def getZFreq(self, Nr):
        Zr = {}
        q = 0
        r = 0
        t = 0
        for i in range(1, len(Nr) - 1):
            q = r
            r = list(Nr.keys())[i]
            t = list(Nr.keys())[i + 1]
            Zr[r] = 2 * Nr[r] / (t - q)
        q = r
        r = t
        Zr[r] = Nr[r] / (r - q)
        return Zr
    
    def estimateNr(self, Zr):
        rList = np.array(list(Zr.keys())).reshape(-1, 1)
        ZrList = np.array(list(Zr.values())).reshape(-1, 1)
        linReg = LinearRegression()
        linReg.fit(np.log(rList), np.log(ZrList))
        rMax = list(Zr.keys())[-1]
        allR = np.arange(1, rMax + 2).reshape(-1, 1)
        return np.exp(linReg.predict(np.log(allR))).reshape(-1)

    def getAdjustedFreq(self, Nr):
        adjustedFreq = np.zeros(len(Nr))
        for i in range(1, len(Nr)):
            if Nr[i] == 0:
                continue
            adjustedFreq[i] = (i + 1) * Nr[i] / Nr[i - 1]
        adjustedFreq[0] = Nr[0]
        return adjustedFreq
    
    def newFreq(self):
        Nr = self.getNr()
        Nr = self.getZFreq(Nr)
        Nr = self.estimateNr(Nr)
        return self.getAdjustedFreq(Nr)
