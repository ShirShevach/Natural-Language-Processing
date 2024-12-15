import spacy
import numpy as np
from datasets import load_dataset
from collections import defaultdict


class Unigram:
    def __init__(self):
        self.words_num = 0
        self.words_dict = defaultdict(int)

    def train(self, texts):
        for text in texts["text"]:
            tokens = nlp(text)
            for token in tokens:
                if token.is_alpha:
                    self.words_dict[token.lemma_] += 1
                    self.words_num += 1

    def probabilityOfWord(self, word):
        return self.words_dict[word] / self.words_num

    def probabilityOfSentence(self, sentence):
        p = 1

        for token in sentence:
            if token.lemma_ in self.words_dict:
                p *= self.probabilityOfWord(token.lemma_)
            else:
                return -np.inf
        return np.log(p)

    def perplexity(self, sentences):
        prob_sum = M = 0
        for sentence in sentences:
            prob_sum += self.probabilityOfSentence(sentence)
            M += len(sentence)
        return np.power(np.e, -(1/M) * prob_sum)


class Bigram:
    def __init__(self):
        self.words_num = 0
        self.words_dict = defaultdict(lambda: defaultdict(int))
        self.words_count = defaultdict(int)

    def train(self, texts):
        for text in texts["text"]:
            tokens = nlp(text)
            prev = "START"
            for token in tokens:
                if token.is_alpha:
                    self.words_dict[prev][token.lemma_] += 1
                    self.words_count[prev] += 1
                    prev = token.lemma_
                    self.words_num += 1

    def probabilityOfWord(self, cur, prev):
        return self.words_dict[prev][cur] / self.words_count[prev]

    def probabilityOfSentence(self, sentence):
        prev = "START"
        p = 1

        for token in sentence:
            if prev in self.words_dict and token.lemma_ in self.words_dict[prev]:
                p *= self.probabilityOfWord(token.lemma_, prev)
                prev = token.lemma_
            else:
                return -np.inf
        return np.log(p)

    def predictNextWord(self, sentence):
        last_word = sentence.split()[-1]
        return max(self.words_dict[last_word], key=self.words_dict[last_word].get)

    def perplexity(self, sentences):
        prob_sum = M = 0
        for sentence in sentences:
            prob_sum += self.probabilityOfSentence(sentence)
            M += len(sentence)
        return np.power(np.e, -(1/M) * prob_sum)


class LIS:
    def __init__(self, delta_uni, delta_bi):
        self.unigram = Unigram()
        self.bigram = Bigram()
        self.delta_uni = delta_uni
        self.delta_bi = delta_bi

    def train(self, texts):
        self.unigram.train(texts)
        self.bigram.train(texts)

    def probabilityOfSentence(self, sentence):
        prev = "START"
        p = 1

        for token in sentence:
            p_bigram = p_unigram = 0

            if prev in self.bigram.words_dict and token.lemma_ in self.bigram.words_dict[prev]:
                p_bigram = self.bigram.probabilityOfWord(token.lemma_, prev)

            if token.lemma_ in self.unigram.words_dict:
                p_unigram = self.unigram.probabilityOfWord(token.lemma_)

            p *= self.delta_uni * p_unigram + self.delta_bi * p_bigram
            prev = token.lemma_

        if p == 0:
            return -np.inf

        return np.log(p)

    def perplexity(self, sentences):
        prob_sum = M = 0
        for sentence in sentences:
            prob_sum += self.probabilityOfSentence(sentence)
            M += len(sentence)
        return np.power(np.e, -(1/M) * prob_sum)


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    unigram = Unigram()
    unigram.train(texts)
    bigram = Bigram()
    bigram.train(texts)

    predicted_word = bigram.predictNextWord("I have a house in")
    print("Q2.\nThe word with the highest probability to come next is:", predicted_word)
    sentences = [nlp("Brad Pitt was born in Oklahoma"), nlp("The actor was born in USA")]
    bi_s1 = bigram.probabilityOfSentence(sentences[0])
    bi_s2 = bigram.probabilityOfSentence(sentences[1])

    print("Q3.a. \nThe probability of the first sentence by Bigram Model is:", bi_s1)
    print("The probability of the second sentence by Bigram Model is:", bi_s2)

    bi_perp = bigram.perplexity(sentences)
    print("Q3.b. \nThe perplexity of the sentences by the Bigram Model is:", bi_perp)

    delta_bi = 2/3
    delta_uni = 1/3
    lis = LIS(delta_uni, delta_bi)
    lis.train(texts)
    lis_s1 = lis.probabilityOfSentence(sentences[0])
    lis_s2 = lis.probabilityOfSentence(sentences[1])
    print("Q4. \nThe probability of the first sentence by LIS Model is:", lis_s1)
    print("The probability of the second sentence by LIS Model is:", lis_s2)
    lis_perp = lis.perplexity(sentences)
    print("The perplexity of the sentences by the LIS Model is:", lis_perp)

