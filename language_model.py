from tokenizer import Tokenizer
from nGrams import NGramModel
import sys
import pickle
import os

n = 3
smoothing = sys.argv[1]
corpusPath = sys.argv[2]
inputText = input("input sentence: ")
tokenizer = Tokenizer()
inputTokenized = tokenizer.tokenize(inputText)
if len(inputTokenized) > 1:
    print("Error: Input has more than one sentence.")
    sys.exit()
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
print("score: ", nGrams.getSentenceScore(inputTokenized[0]))
