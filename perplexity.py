from tokenizer import Tokenizer
from nGrams import NGramModel
from sklearn.model_selection import train_test_split
import pickle
import os

tokenizer = Tokenizer()
with open('./data/PrideandPrejudice.txt', 'r', encoding='utf8') as file:
    textP = file.read()
with open('./data/Ulysses.txt', 'r', encoding='utf8') as file:
    textU = file.read()
tokenizedP = tokenizer.tokenize(textP)
tokenizedU = tokenizer.tokenize(textU)

trainSetP, testSetP = train_test_split(tokenizedP, test_size=1000/len(tokenizedP), random_state=29)
trainSetU, testSetU = train_test_split(tokenizedU, test_size=1000/len(tokenizedU), random_state=29)

start = './2021111029_'
end = '.pkl'

if os.path.exists(start + 'LM1' + end):
    with open(start + 'LM1' + end, 'rb') as file:
        LM1 = pickle.load(file)
else:
    LM1 = NGramModel(3, 'g')
    LM1.fit(trainSetP)
    with open(start + 'LM1' + end, 'wb') as file:
        pickle.dump(LM1, file)

if os.path.exists(start + 'LM2' + end):
    with open(start + 'LM2' + end, 'rb') as file:
        LM2 = pickle.load(file)
else:
    LM2 = NGramModel(3, 'i')
    LM2.fit(trainSetP)
    with open(start + 'LM2' + end, 'wb') as file:
        pickle.dump(LM2, file)

if os.path.exists(start + 'LM3' + end):
    with open(start + 'LM3' + end, 'rb') as file:
        LM3 = pickle.load(file)
else:
    LM3 = NGramModel(3, 'g')
    LM3.fit(trainSetU)
    with open(start + 'LM3' + end, 'wb') as file:
        pickle.dump(LM3, file)

if os.path.exists(start + 'LM4' + end):
    with open(start + 'LM4' + end, 'rb') as file:
        LM4 = pickle.load(file)
else:
    LM4 = NGramModel(3, 'i')
    LM4.fit(trainSetU)
    with open(start + 'LM4' + end, 'wb') as file:
        pickle.dump(LM4, file)

if __name__ == "__main__":
    avgPerp, sentencePerp = LM1.perplexity(trainSetP)
    with open('2021111029_LM1_train-perplexity.txt', 'w', encoding='utf8') as file:
        file.write(str(avgPerp) + "\n")
        for i in sentencePerp.keys():
            file.write(i + "\t" + str(sentencePerp[i]) + "\n")

    avgPerp, sentencePerp = LM1.perplexity(testSetP)
    with open('2021111029_LM1_test-perplexity.txt', 'w', encoding='utf8') as file:
        file.write(str(avgPerp) + "\n")
        for i in sentencePerp.keys():
            file.write(i + "\t" + str(sentencePerp[i]) + "\n")

    avgPerp, sentencePerp = LM2.perplexity(trainSetP)
    with open('2021111029_LM2_train-perplexity.txt', 'w', encoding='utf8') as file:
        file.write(str(avgPerp) + "\n")
        for i in sentencePerp.keys():
            file.write(i + "\t" + str(sentencePerp[i]) + "\n")

    avgPerp, sentencePerp = LM2.perplexity(testSetP)
    with open('2021111029_LM2_test-perplexity.txt', 'w', encoding='utf8') as file:
        file.write(str(avgPerp) + "\n")
        for i in sentencePerp.keys():
            file.write(i + "\t" + str(sentencePerp[i]) + "\n")

    avgPerp, sentencePerp = LM3.perplexity(trainSetU)
    with open('2021111029_LM3_train-perplexity.txt', 'w', encoding='utf8') as file:
        file.write(str(avgPerp) + "\n")
        for i in sentencePerp.keys():
            file.write(i + "\t" + str(sentencePerp[i]) + "\n")

    avgPerp, sentencePerp = LM3.perplexity(testSetU)
    with open('2021111029_LM3_test-perplexity.txt', 'w', encoding='utf8') as file:
        file.write(str(avgPerp) + "\n")
        for i in sentencePerp.keys():
            file.write(i + "\t" + str(sentencePerp[i]) + "\n")
            
    avgPerp, sentencePerp = LM4.perplexity(trainSetU)
    with open('2021111029_LM4_train-perplexity.txt', 'w', encoding='utf8') as file:
        file.write(str(avgPerp) + "\n")
        for i in sentencePerp.keys():
            file.write(i + "\t" + str(sentencePerp[i]) + "\n")

    avgPerp, sentencePerp = LM4.perplexity(testSetU)
    with open('2021111029_LM4_test-perplexity.txt', 'w', encoding='utf8') as file:
        file.write(str(avgPerp) + "\n")
        for i in sentencePerp.keys():
            file.write(i + "\t" + str(sentencePerp[i]) + "\n")