from tokenizer import Tokenizer
from nGrams import NGramModel
from sklearn.model_selection import train_test_split
import sys

arglen = len(sys.argv)
n = 3
smoothing = sys.argv[1]
corpusPath = sys.argv[2]
inputText = input("input sentence: ")
tokenizer = Tokenizer()
inputTokenized = tokenizer.tokenize(inputText)
if len(inputTokenized) > 1:
    print("Error: Input has more than one sentence.")
    sys.exit()
nGrams = NGramModel(n, smoothing)
with open(corpusPath, 'r', encoding='utf8') as file:
    text = file.read()
tokenized = tokenizer.tokenize(text)
nGrams.fit(tokenized)
print("score: ", nGrams.getSentenceScore(inputTokenized[0]))
