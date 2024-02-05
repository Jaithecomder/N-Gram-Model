from tokenizer import Tokenizer
from nGrams import NGramModel
import sys
import numpy as np

n = 3
smoothing = sys.argv[1]
corpusPath = sys.argv[2]
inputText = input("input sentence: ")
tokenizer = Tokenizer()
inputTokenized = tokenizer.tokenize(inputText)
inputTokenized = np.array(inputTokenized).reshape(-1)
inputTokenized = inputTokenized[:-1]
nGrams = NGramModel(n, smoothing)
with open(corpusPath, 'r', encoding='utf8') as file:
    text = file.read()
tokenized = tokenizer.tokenize(text)
nGrams.fit(tokenized)
words, probs = nGrams.generate(inputTokenized)
for i in range(len(words)):
    print(words[i], probs[i])