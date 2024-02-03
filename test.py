from tokenizer import Tokenizer
from nGrams import NGramModel
from sklearn.model_selection import train_test_split
import random

tokenizer = Tokenizer()
nGrams = NGramModel(3, 'g')
# text = '''Is that what you mean?? <> ab.bc <URL> I am unsure what you mean. www@www.com https://www.dsfsd.com/d._jaflkja#lkdsa12376178246587jlkajs@slkadjlaksjd/oki.html
# 1231213.123124242 100,000'''
with open('./data/PrideandPrejudice.txt', 'r') as file:  
    text = file.read() 
tokenized = tokenizer.tokenize(text)

tokenized, testSet = train_test_split(tokenized, test_size=1000/len(tokenized), random_state=29)
nGrams.fit(tokenized)

# tokens = ['It', 'is']
# print(tokens[0], end=' ')
# print(tokens[1], end=' ')
# tokens.append(nGrams.generate(tokens))
# print(tokens[-1], end=' ')
# for i in range(100):
#     newToken = nGrams.generate(tokens)
#     if newToken == None:
#         break
#     tokens.append(newToken)
#     print(tokens[-1], end=' ')
if __name__ == "__main__":
    print(nGrams.perplexity(testSet))
    print(nGrams.perplexity(tokenized))