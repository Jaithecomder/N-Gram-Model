from tokenizer import Tokenizer
from nGrams import NGramModel
from sklearn.model_selection import train_test_split

tokenizer = Tokenizer()
nGrams = NGramModel(3, 'g')
with open('./data/PrideandPrejudice.txt', 'r', encoding='utf8') as file:
# with open('./data/Ulysses.txt', 'r', encoding='utf8') as file:
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
    print("Test set perplexity : ", nGrams.perplexity(testSet))
    print("Train set perplexity : ", nGrams.perplexity(tokenized))