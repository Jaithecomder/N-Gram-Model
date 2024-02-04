import numpy as np

class LinearInterpolation:
    def __init__(self, nGramsDictList, nGramsProbList, corpusSize, n):
        self.nGramsDictList = nGramsDictList
        self.nGramsProbList = nGramsProbList
        self.corpusSize = corpusSize
        self.n = n
        self.weights = [0] * n

    def learnWeights(self):
        for nGram in self.nGramsDictList[self.n - 1]:
            cases = []
            for i in range(self.n - 1):
                if (self.nGramsDictList[len(nGram[i:-1]) - 1][nGram[i:-1]] - 1) == 0:
                    cases.append(0)
                    continue
                cases.append((self.nGramsDictList[len(nGram[i:]) - 1][nGram[i:]] - 1) / (self.nGramsDictList[len(nGram[i:-1]) - 1][nGram[i:-1]] - 1))
            cases.append((self.nGramsDictList[0][nGram[-1:]] - 1) / (self.corpusSize - 1))
            ind = np.argmax(cases)
            self.weights[ind] += self.nGramsDictList[len(nGram) - 1][nGram]
        sumWeights = sum(self.weights)
        for i in range(self.n):
            self.weights[i] /= sumWeights

    def getProbability(self, nGram):
        prob = 0
        for i in range(self.n - 1):
            if nGram[i:] not in self.nGramsProbList[len(nGram[i:]) - 1]:
                continue
            prob += self.weights[i] * self.nGramsProbList[len(nGram[i:]) - 1][nGram[i:]]
        if nGram[-1:] in self.nGramsProbList[len(nGram[i:]) - 1]:
            prob += self.weights[-1] * self.nGramsProbList[0][nGram[-1:]]
        return prob