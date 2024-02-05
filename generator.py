from tokenizer import Tokenizer
from nGrams import NGramModel
import sys
import numpy as np
import pickle
import os

n = 3
smoothing = sys.argv[1]
corpusPath = sys.argv[2]
inputText = input("input sentence: ")
tokenizer = Tokenizer()
inputTokenized = tokenizer.tokenize(inputText)
inputTokenized = np.array(inputTokenized).reshape(-1)
inputTokenized = inputTokenized[:-1]
start = './2021111029_'
end = '.pkl'
if smoothing == 'g' and corpusPath == '.\\data\\PrideandPrejudice.txt':
    start += 'LM1'
elif smoothing == 'i' and corpusPath == '.\\data\\PrideandPrejudice.txt':
    start += 'LM2'
elif smoothing == 'g' and corpusPath == '.\\data\\Ulysses.txt':
    start += 'LM3'
elif smoothing == 'i' and corpusPath == '.\\data\\Ulysses.txt':
    start += 'LM4'
if os.path.exists(start + end):
    with open(start + end, 'rb') as file:
        nGrams = pickle.load(file)
else:
    nGrams = NGramModel(n, smoothing)
    with open(corpusPath, 'r', encoding='utf8') as file:
        text = file.read()
    tokenized = tokenizer.tokenize(text)
    nGrams.fit(tokenized)
    with open(start + end, 'wb') as file:
        pickle.dump(nGrams, file)
words, probs = nGrams.generate(inputTokenized)
for i in range(len(words)):
    print(words[i], probs[i])