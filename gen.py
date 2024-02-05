from tokenizer import Tokenizer
from nGrams import NGramModel
from sklearn.model_selection import train_test_split

tokenizer = Tokenizer()
nGrams = NGramModel(3, 'i')
# with open('./data/PrideandPrejudice.txt', 'r', encoding='utf8') as file:
with open('./data/Ulysses.txt', 'r', encoding='utf8') as file:
    text = file.read() 
tokenized = tokenizer.tokenize(text)

tokenized, testSet = train_test_split(tokenized, test_size=1000/len(tokenized))
nGrams.fit(tokenized)

tokens = ['<s>']
print(tokens[0], end=" ")
for i in range(0, 100):
    newToken = nGrams.genTokens(tokens)
    tokens.append(newToken)
    print(newToken, end=" ")