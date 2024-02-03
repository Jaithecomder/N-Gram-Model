import re

class Tokenizer:
    def replaceAngularBrackets(self, text):
        text = text.replace('<', '\<')
        text = text.replace('>', '\>')
        return text

    def replaceEmail(self, text):
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '<MAILID>', text)
        return text

    def replaceURL(self, text):
        text = re.sub(r'(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '<URL>', text)
        return text
    
    def replaceNumber(self, text):
        text = re.sub(r'\d+', '<NUM>', text)
        return text
    
    def replaceHashtags(self, text):
        text = re.sub(r'#\w+', '<HASHTAG>', text)
        return text
    
    def replaceMentions(self, text):
        text = re.sub(r'@\w+', '<MENTION>', text)
        return text
    
    def sentenceTokenize(self, text):
        text = re.split(r'([.!?]+)', text)
        text = list(filter(None, text))
        combinedText = []
        for i in range(0, len(text), 2):
            if i+1 < len(text):
                combinedText.append(text[i] + text[i+1])
        if len(text) % 2 != 0:
            combinedText.append(text[-1])
        return combinedText
    
    def wordTokenize(self, text):
        text = re.split(r'\s|\\(<)|\\(>)|([^\w<>])', text)
        text = list(filter(None, text))
        return text

    def tokenize(self, text):
        text = text.replace('\n', ' ')
        text = self.replaceAngularBrackets(text)
        text = self.replaceEmail(text)
        text = self.replaceURL(text)
        text = self.replaceNumber(text)
        text = self.replaceHashtags(text)
        text = self.replaceMentions(text)

        text = self.sentenceTokenize(text)
        tokenizedText = []
        for sentence in text:
            tokenizedSentence = self.wordTokenize(sentence)
            if len(tokenizedSentence) > 0:
                tokenizedSentence.insert(0, '<s>')
                tokenizedSentence.append('</s>')
                tokenizedText.append(tokenizedSentence)
        return tokenizedText