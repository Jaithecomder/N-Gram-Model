import numpy as np

class LinearInterpolation:
    def __init__(self, nGramsDict, nGramsProb, corpusSize, n):
        self.nGramsDict = nGramsDict
        self.nGramsProb = nGramsProb
        self.corpusSize = corpusSize
        self.n = n
        self.weights = [0] * n

    def learnWeights(self):
        for nGram in self.nGramsDict:
            if len(nGram) != self.n:
                continue
            cases = []
            for i in range(self.n - 1):
                if (self.nGramsDict[nGram[i:-1]] - 1) == 0:
                    cases.append(0)
                    continue
                cases.append((self.nGramsDict[nGram[i:]] - 1) / (self.nGramsDict[nGram[i:-1]] - 1))
            cases.append((self.nGramsDict[nGram[-1:]] - 1) / (self.corpusSize - 1))
            ind = np.argmax(cases)
            self.weights[ind] += self.nGramsDict[nGram]
        sumWeights = sum(self.weights)
        for i in range(self.n):
            self.weights[i] /= sumWeights

    def getProbability(self, nGram):
        prob = 0
        for i in range(self.n - 1):
            if nGram[i:] not in self.nGramsProb:
                continue
            prob += self.weights[i] * self.nGramsProb[nGram[i:]]
        if nGram[-1:] in self.nGramsProb:
            prob += self.weights[-1] * self.nGramsProb[nGram[-1:]]
        return prob