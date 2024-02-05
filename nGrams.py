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
        self.corpusSize = 0
        self.vocab = set()
        self.zeroCounts = []

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

        for i in range(1, self.n):
            for nGram in nGramsDictList[i]:
                if nGram[:-1] not in nGramSumList[len(nGram) - 2]:
                    nGramSumList[len(nGram) - 2][nGram[:-1]] = 0
                nGramSumList[len(nGram) - 2][nGram[:-1]] += nGramsDictList[len(nGram) - 1][nGram]

        for nGram in nGramsDictList[0]:
            denm = self.corpusSize
            if self.fn == 'g':
                denm += self.zeroCounts[0]
            nGramsProbList[0][nGram] = nGramsDictList[0][nGram] / denm

        for i in range(1, self.n):
            for nGram in nGramsDictList[i]:
                denm = nGramSumList[len(nGram) - 2][nGram[:-1]]
                if self.fn == 'g':
                    denm += self.zeroCounts[i]
                nGramsProbList[len(nGram) - 1][nGram] = nGramsDictList[len(nGram) - 1][nGram] / denm
        if self.fn == 'g':
            nGramsProbList[0]['<0>'] = self.zeroCounts[0] / (self.corpusSize + self.zeroCounts[0])
            for i in range(0, self.n - 1):
                for nM1Gram in nGramSumList[i]:
                    denm = nGramSumList[i][nM1Gram] + self.zeroCounts[i + 1]
                    nGramsProbList[i + 1][nM1Gram + ('<0>',)] = self.zeroCounts[i + 1] / denm
        return nGramsProbList
    
    def getProbability(self, nGram):
        if self.fn == 'i':
            prob = self.linInt.getProbability(nGram)
            if prob == 0:
                prob = 1e-8
            return prob

        if nGram in self.nGramsDictList[len(nGram) - 1]:
            return self.nGramsProbList[len(nGram) - 1][nGram]
        
        # if nGram[:-1] in self.nGramsDictList:
        #     return self.nGramsProbList[len(nGram) - 1][nGram[:-1] + ('<0>',)]
        
        return 1e-8

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
        self.zeroCounts = []
        for i in range(1, self.n + 1):
            nGramsDictNew = {}
            gt = GoodTuring(i, self.nGramsDictList[i-1])
            newFreq = gt.newFreq()
            for nGram in self.nGramsDictList[i-1]:
                nGramsDictNew[nGram] = newFreq[self.nGramsDictList[i-1][nGram]]
            self.zeroCounts.append(newFreq[0])
            self.nGramsDictList[i-1] = nGramsDictNew
    
    def fit(self, text):
        self.corpusSize = sum([len(sentence) for sentence in text])
        self.vocab = self.getVocab(text)
        self.nGramsDictList = self.getFrequency(text)
        if self.fn == 'g':
            self.goodTuring()
        self.nGramsProbList = self.computeProbability(self.nGramsDictList)
        if self.fn == 'i':
            self.linInt = LinearInterpolation(self.nGramsDictList, self.nGramsProbList, self.corpusSize, self.n)
            self.linInt.learnWeights()

    def generate(self, tokens):
        words = [''] * 5
        probs = [0] * 5
        context = tokens
        if len(tokens) >= self.n:
            context = tokens[-self.n + 1:]
        count = 0
        for nGram in self.nGramsDictList[len(context)]:
            if nGram[:-1] == tuple(context):
                count += 1
                prob = self.getProbability(nGram)
                if prob > min(probs):
                    ind = probs.index(min(probs))
                    words[ind] = nGram[-1]
                    probs[ind] = prob
        sortedInds = np.argsort(probs)[::-1]
        words = [words[i] for i in sortedInds]
        probs = [probs[i] for i in sortedInds]
        if count < 5:
            words = words[:count]
            probs = probs[:count]
        return words, probs
    
    def perplexity(self, testSet):
        numSentences = len(testSet)
        procs = 4
        testSet = [testSet[i * numSentences // procs: (i + 1) * numSentences // procs] for i in range(procs)]
        perplexity = []
        # with ProcessPoolExecutor(max_workers=procs) as executor:
        #     futures = {executor.submit(self.getPerplexity, sentenceSet): sentenceSet for sentenceSet in testSet}
        #     for future in as_completed(futures):
        #         perplexity.append(future.result())
        for sentenceSet in testSet:
            perplexity.append(self.getPerplexity(sentenceSet))
        avgPerp = 0
        sentencePerp = {}
        for i in perplexity:
            avgPerp += sum(i.values())
            sentencePerp.update(i)
        avgPerp /= numSentences
        return avgPerp, sentencePerp
    
    def getPerplexity(self, sentenceSet):
        perpDict = {}
        for sentence in sentenceSet:
            nList = []
            logProb = 0
            for word in sentence:
                nList.append(word)
                if len(nList) == self.n + 1:
                    nList.pop(0)
                logProb += np.log(self.getProbability(tuple(nList)))
            prob = np.exp(logProb)
            if prob == 0:
                prob = 1e-10
            perpDict[' '.join(sentence)] = prob ** (-1 / len(sentence))
        return perpDict
    
    def getSentenceScore(self, sentence):
        nList = []
        logProb = 0
        for word in sentence:
            nList.append(word)
            if len(nList) == self.n + 1:
                nList.pop(0)
            logProb += np.log(self.getProbability(tuple(nList)))
        return np.exp(logProb)