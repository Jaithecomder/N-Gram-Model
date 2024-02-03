import random
import numpy as np
from goodTuring import GoodTuring
from linearInterpolation import LinearInterpolation
from concurrent.futures import ProcessPoolExecutor, as_completed

class NGramModel:
    def __init__(self, n=3, fn='g'):
        self.n = n
        self.fn = fn
        self.nGramsDict = {}
        self.linInt = None

    def getVocab(self, text):
        vocab = set()
        for sentence in text:
            for word in sentence:
                vocab.add(word)
        return vocab
    
    def getFrequency(self, text):
        nGramsDict = {}
        nList = []
        for sentence in text:
            for word in sentence:
                nList.append(word)
                if len(nList) == self.n + 1:
                    nList.pop(0)
                for i in range(len(nList)):
                    nGram = tuple(nList[i:])
                    if nGram in nGramsDict:
                        nGramsDict[nGram] += 1
                    else:
                        nGramsDict[nGram] = 1
        return nGramsDict
    
    def computeProbability(self, nGramsDict):
        nGramsProb = {}
        nGramSum = {}
        for nGram in nGramsDict:
            if type(nGram) == int:
                continue
            if nGram[:-1] not in nGramSum:
                nGramSum[nGram[:-1]] = 0
            nGramSum[nGram[:-1]] += nGramsDict[nGram]
        for nGram in nGramsDict:
            if type(nGram) == int:
                if nGram == 1:
                    nGramsProb[nGram] = nGramsDict[nGram] / self.corpusSize
                continue
            if len(nGram) == 1:
                nGramsProb[nGram] = nGramsDict[nGram] / self.corpusSize
            nGramsProb[nGram] = nGramsDict[nGram] / nGramSum[nGram[:-1]]
        return nGramsProb
    
    def getProbability(self, nGram):
        if self.fn == 'l':
            prob = self.linInt.getProbability(nGram)
            if prob == 0:
                prob = 1e-10
            return prob
        
        if nGram in self.nGramsDict:
            return self.nGramsProb[nGram]

        if len(nGram) == 1:
            return 1e-5
        
        if nGram[-1:] not in self.nGramsDict:
            return 1e-5
        
        if nGram[:-1] not in self.nGramsDict:
            return self.nGramsProb[nGram[-1:]]
        
        if nGram[:-1] in self.nGramsDict:
            nSum = 0
            for nGramKey in self.nGramsDict:
                if type(nGramKey) == int:
                    continue
                if nGramKey[:-1] == tuple(nGram[:-1]):
                    nSum += self.nGramsDict[nGramKey]
            return self.nGramsDict[len(nGram)] / nSum

        return 1e-5

    def goodTuring(self, nGramsDict):
        nGramsDictNew = {}
        for i in range(1, self.n + 1):
            gt = GoodTuring(i, nGramsDict)
            newFreq = gt.newFreq()
            for nGram in nGramsDict:
                if len(nGram) != i:
                    continue
                nGramsDictNew[nGram] = newFreq[nGramsDict[nGram]]
            for i in range(1, self.n + 1):
                if i not in nGramsDictNew:
                    nGramsDictNew[i] = newFreq[0]
        return nGramsDictNew        
    
    def fit(self, text):
        self.corpusSize = sum([len(sentence) for sentence in text])
        self.vocab = self.getVocab(text)
        self.nGramsDict = self.getFrequency(text)
        if self.fn == 'g':
            self.nGramsDict = self.goodTuring(self.nGramsDict)
        self.nGramsProb = self.computeProbability(self.nGramsDict)
        if self.fn == 'l':
            self.linInt = LinearInterpolation(self.nGramsDict, self.nGramsProb, self.corpusSize, self.n)
            self.linInt.learnWeights()

    def generate(self, tokens):
        lastNM1Gram = tuple(tokens[-(self.n-1):])
        if lastNM1Gram not in self.nGramsDict:
            return None
        nGramsProb = self.getProbability(self.nGramsDict, lastNM1Gram)
        cumProb = 0
        predWord = ''
        rand = random.random()
        for nGram in nGramsProb:
            if nGram[:-1] == lastNM1Gram:
                cumProb += nGramsProb[nGram]
                if cumProb >= rand:
                    predWord = nGram[-1]
                    break
        return predWord
    
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
                prob = 1e-5
            perpSum += prob ** (-1/len(sentence))
        return perpSum