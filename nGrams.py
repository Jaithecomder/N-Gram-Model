import random
import numpy as np
from goodTuring import GoodTuring
from linearInterpolation import LinearInterpolation
from concurrent.futures import ProcessPoolExecutor, as_completed

class NGramModel:
    def __init__(self, n=3, fn='g'):
        self.n = n
        self.fn = fn
        self.nGramsDictList = {}
        self.linInt = None

    def getVocab(self, text):
        vocab = set()
        for sentence in text:
            for word in sentence:
                vocab.add(word)
        return vocab
    
    def getFrequency(self, text):
        freqList = []
        for i in range(0, self.n):
            freqList.append({})
        nList = []
        for sentence in text:
            for word in sentence:
                nList.append(word)
                if len(nList) == self.n + 1:
                    nList.pop(0)
                for i in range(len(nList)):
                    nGram = tuple(nList[i:])
                    if nGram in freqList[len(nGram) - 1]:
                        freqList[len(nGram) - 1][nGram] += 1
                    else:
                        freqList[len(nGram) - 1][nGram] = 1
        return freqList
    
    def computeProbability(self, nGramsDictList):
        nGramsProbList = []
        nGramSumList = []
        for i in range(0, self.n):
            nGramsProbList.append({})

        for i in range(0, self.n - 1):
            nGramSumList.append({})

        for i in range(0, self.n):
            for nGram in nGramsDictList[i]:
                if len(nGram) == 1:
                    continue
                if nGram[:-1] not in nGramSumList[len(nGram) - 2]:
                    nGramSumList[len(nGram) - 2][nGram[:-1]] = 0
                nGramSumList[len(nGram) - 2][nGram[:-1]] += nGramsDictList[len(nGram) - 1][nGram]

        for i in range(0, self.n):
            for nGram in nGramsDictList[i]:
                if len(nGram) == 1:
                    nGramsProbList[0][nGram] = nGramsDictList[0][nGram] / self.corpusSize
                    continue
                nGramsProbList[len(nGram) - 1][nGram] = nGramsDictList[len(nGram) - 1][nGram] / nGramSumList[len(nGram) - 2][nGram[:-1]]
        return nGramsProbList
    
    def getProbability(self, nGram):
        if self.fn == 'l':
            prob = self.linInt.getProbability(nGram)
            if prob == 0:
                prob = 1e-5
            return prob
        
        if nGram in self.nGramsDictList:
            return self.nGramsProbList[nGram]

        if len(nGram) == 1:
            return 1e-10
        
        if nGram[-1:] not in self.nGramsDictList:
            return 1e-10
        
        if nGram[:-1] not in self.nGramsDictList:
            return self.nGramsProbList[nGram[-1:]]
        
        if nGram[:-1] in self.nGramsDictList:
            nSum = 0
            for nGramKey in self.nGramsDictList:
                if type(nGramKey) == int:
                    continue
                if nGramKey[:-1] == tuple(nGram[:-1]):
                    nSum += self.nGramsDictList[nGramKey]
            return self.nGramsDictList[len(nGram)] / nSum

        return 1e-5

    def goodTuring(self):
        nGramsDictNew = {}
        for i in range(1, self.n + 1):
            gt = GoodTuring(i, self.nGramsDictList[i-1])
            newFreq = gt.newFreq()
            for nGram in self.nGramsDictList[i-1]:
                nGramsDictNew[nGram] = newFreq[self.nGramsDictList[i-1][nGram]]
            self.nGramsDictList[i-1] = nGramsDictNew       
    
    def fit(self, text):
        self.corpusSize = sum([len(sentence) for sentence in text])
        self.vocab = self.getVocab(text)
        self.nGramsDictList = self.getFrequency(text)
        if self.fn == 'g':
            self.nGramsDictList = self.goodTuring()
        self.nGramsProbList = self.computeProbability(self.nGramsDictList)
        if self.fn == 'l':
            self.linInt = LinearInterpolation(self.nGramsDictList, self.nGramsProbList, self.corpusSize, self.n)
            self.linInt.learnWeights()

    # def generate(self, tokens):
    #     lastNM1Gram = tuple(tokens[-(self.n-1):])
    #     if lastNM1Gram not in self.nGramsDictList:
    #         return None
    #     nGramsProbList = self.getProbability(self.nGramsDictList, lastNM1Gram)
    #     cumProb = 0
    #     predWord = ''
    #     rand = random.random()
    #     for nGram in nGramsProbList:
    #         if nGram[:-1] == lastNM1Gram:
    #             cumProb += nGramsProbList[nGram]
    #             if cumProb >= rand:
    #                 predWord = nGram[-1]
    #                 break
    #     return predWord
    
    def perplexity(self, testSet):
        numSentences = len(testSet)
        procs = 4
        testSet = [testSet[i * numSentences // procs: (i + 1) * numSentences // procs] for i in range(procs)]
        perplexity = []
        with ProcessPoolExecutor(max_workers=procs) as executor:
            futures = {executor.submit(self.getPerplexity, sentenceSet): sentenceSet for sentenceSet in testSet}
            for future in as_completed(futures):
                perplexity.append(future.result())
        # for sentenceSet in testSet:
        #     perplexity.append(self.getPerplexity(sentenceSet))
        return sum(perplexity) / numSentences
    
    def getPerplexity(self, sentenceSet):
        perpSum = 0
        for sentence in sentenceSet:
            nList = []
            logProb = 0
            for word in sentence:
                nList.append(word)
                if len(nList) == self.n + 1:
                    nList.pop(0)
                logProb += np.log(self.getProbability(tuple(nList)))
            # if prob == 0:
            #     prob = 1e-10
            prob = np.exp(logProb)
            if prob == 0:
                prob = 1e-10
            perpSum += prob ** (-1/len(sentence))
        return perpSum