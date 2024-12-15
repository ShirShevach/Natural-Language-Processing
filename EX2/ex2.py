import nltk
from nltk.corpus import brown
from collections import defaultdict
import numpy as np


def clean_tag(tag):
    """Removes the first '+', '-', or '*' from the string, if present."""
    if tag == "--":  # I add, not sure this right
        return tag
    tag = tag.split("+", 1)[0]
    tag = tag.split("-", 1)[0]
    tag = tag.split("$", 1)[0]  # I add!!
    return tag.split("*", 1)[0]


def make_pseudo_words(word):
    """Assigns a pseudo words based on word characteristics."""
    verb_suffixes = ['ate', 'ify', 'ize', 'ise', 'en', 'ed', 'ing']
    for suffix in verb_suffixes:
        if word.endswith(suffix):
            return "VERB"
    if word.startswith("$"):
        return "MONEY"
    if word.endswith("'s") or word.endswith("s'"):
        return "POSSESSION"
    if (word.endswith("th") or word.endswith("st") or word.endswith("nd") or word.endswith("rd")) \
            and word[:-2].isdigit():
        return "ORDINAL"
    if word.endswith("ly") or word.endswith("ward") or word.endswith("wise"):
        return "ADV"
    adjective_suffixes = ['able', 'ible', 'al', 'ful', 'ic', 'ous', 'ive', 'less']
    adjective_prefixes = ['un', 'dis', 'in', 'im', 'il', 'ir']
    for suffix in adjective_suffixes:
        if word.endswith(suffix):
            return "ADJ"
    for prefix in adjective_prefixes:
        if word.startswith(prefix):
            return "ADJ"
    if word.isdigit():
        if (word.startswith("19") or word.startswith("20")) and len(word) == 4:
            return "YEAR"
        else:
            return "DIGIT"
    noun_suffixes = ['tion', 'sion', 'ment', 'ness', 'ity', 'ism', 'hood', 'cy', 'ance', 'ence']
    for suffix in noun_suffixes:
        if word.endswith(suffix):
            return "NOUN"
    if word[0].isupper():
        return "CAPITAL"
    return "OTHER"


class TagBaseline:
    def __init__(self):
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.tag_count = defaultdict(int)
        self.max_tags = defaultdict(int)

    def train(self, train_set):
        """Trains the baseline tagger using the training set."""
        for sentence in train_set:
            for word, tag in sentence:
                tag = clean_tag(tag)
                self.word_counts[word][tag] += 1
                self.tag_count[tag] += 1

        # Searches the tag with maximum likelihood for every word
        for word, tag_dict in self.word_counts.items():
            max_tag = max(tag_dict, key=tag_dict.get)
            self.max_tags[word] = max_tag

    def error_rates(self, test_set):
        """Calculates error rates for known and unknown words."""
        known = unknown = total_known = total_unknown = 0
        for sentence in test_set:
            for word, true_tag in sentence:
                true_tag = clean_tag(true_tag)
                if word not in self.max_tags:
                    tag = 'NN'
                    total_unknown += 1
                else:
                    tag = self.max_tags[word]
                    total_known += 1

                if tag == true_tag:
                    if word in self.max_tags:
                        known += 1

                    else:
                        unknown += 1

        known_error_rate = 1 - (known / total_known)
        unknown_error_rate = 1 - (unknown / total_unknown)
        total_error_rate = 1 - ((known + unknown) / (total_known + total_unknown))

        return known_error_rate, unknown_error_rate, total_error_rate


class HMM:
    def __init__(self):
        self.tagBaseline = TagBaseline()
        self.transition = defaultdict(lambda: defaultdict(int))
        self.tag_count = defaultdict(int)
        self.emission = defaultdict(lambda: defaultdict(int))
        self.words = []
        self.words_count = defaultdict(int)

    def train(self, train_set, pseudo_words=False):
        """Trains the Hidden Markov Model."""
        if pseudo_words:
            train_set = self.split_vocabulary(train_set)
        tagBaseline.train(train_set)
        prev = "START"
        for sentence in train_set:
            for word, tag in sentence:
                tag = clean_tag(tag)
                self.transition[prev][tag] += 1
                self.tag_count[prev] += 1
                self.emission[word][tag] += 1
                self.words.append(word)
                prev = tag

                if word == sentence[-1][0]:  # if word is the last in sentence
                    self.transition[tag]["STOP"] += 1

    def split_vocabulary(self, set):
        """Replaces infrequent words with pseudo words."""
        for sentence in set:
            for (word, tag) in sentence:
                self.words_count[word] += 1
        new_set = []
        for sentence in set:
            new_set.append([])
            for i, (word, tag) in enumerate(sentence):
                if self.words_count[word] < 5:
                    word = make_pseudo_words(word)
                new_set[-1].append((word, tag))
        return new_set

    def e(self, word, tag, laplace_smooth):
        """Calculate emission probability."""
        if word not in self.emission:
            if tag == "NN":
                emission_prob = (1 + laplace_smooth) / (self.tag_count[tag] + laplace_smooth * len(self.words))
            else:
                emission_prob = (0 + laplace_smooth) / (self.tag_count[tag] + laplace_smooth * len(self.words))
        else:
            emission_prob = (self.emission[word][tag] + laplace_smooth) / \
                            (self.tag_count[tag] + laplace_smooth * len(self.words))
        return emission_prob

    def q(self, current, prev):
        """Calculate transition probability."""
        if prev in self.transition and current in self.transition[prev] and prev in self.tag_count:
            return self.transition[prev][current] / self.tag_count[prev]
        return 0.00000000000001 / len(self.tag_count)

    def viterbi(self, x, laplace_smooth):
        """Implement the Viterbi algorithm for sequence labeling."""
        tags = list(self.tag_count.keys())
        pi = dict()
        bp = dict()
        n = len(x)
        for tag in tags:
            pi[(-1, tag)] = 0
        pi[(-1, "START")] = 1
        for k in range(0, n):
            for curr_tag in tags:
                arr = [pi[(k - 1, prev_tag)] * self.q(curr_tag, prev_tag)
                       * self.e(x[k], curr_tag, laplace_smooth)
                       for prev_tag in tags]
                pi[(k, curr_tag)] = max(arr)
                bp[(k, curr_tag)] = np.argmax(arr)
        y = ["" for _ in range(n)]
        y[n - 1] = tags[np.argmax([pi[(n - 1, tag)] * self.q("STOP", tag) for tag in tags])]
        for k in range(n - 2, -1, -1):
            y[k] = tags[bp[(k + 1, y[k + 1])]]
        return y

    def error_rate(self, test_set, laplace_smooth=0, pseudo_words=False):
        """Calculate error rates and confusion matrix."""
        known = unknown = total_known = total_unknown = 0
        if pseudo_words:
            test_set = self.split_vocabulary(test_set)

        # make confusion matrix
        confusion_matrix = defaultdict(int)

        for sentence in test_set:
            words_sentence = [word for word, tag in sentence]
            best_path_tag_name = self.viterbi(words_sentence, laplace_smooth)
            for i, (word, true_tag) in enumerate(sentence):
                true_tag = clean_tag(true_tag)
                predicted_tag = best_path_tag_name[i]
                if word in self.words:
                    total_known += 1
                    if predicted_tag == true_tag:
                        known += 1
                    else:
                        confusion_matrix[(true_tag, predicted_tag)] += 1
                else:
                    total_unknown += 1
                    if predicted_tag == true_tag:
                        unknown += 1
                    else:
                        confusion_matrix[(true_tag, predicted_tag)] += 1

        known_error_rate = 1 - (known / total_known)
        unknown_error_rate = 1 - (unknown / total_unknown)
        total_error_rate = 1 - ((known + unknown) / (total_known + total_unknown))

        return known_error_rate, unknown_error_rate, total_error_rate, confusion_matrix


if __name__ == '__main__':
    nltk.download('brown')
    brown_news = brown.tagged_sents(categories='news')
    # Get the total number of sentences
    total_sentences = len(brown_news)

    # Calculate the number of sentences for the test set (10%)
    test_size = int(0.1 * total_sentences)

    # Divide the corpus into training and test sets
    train_set = brown_news[:total_sentences - test_size]
    test_set = brown_news[-test_size:]

    tagBaseline = TagBaseline()
    tagBaseline.train(train_set)
    known_error, unknown_error, total_error = tagBaseline.error_rates(test_set)
    print("B2")
    print("Known word error rate:", known_error)
    print("Unknown word error rate:", unknown_error)
    print("Total error rate:", total_error)

    print("C")
    hmm = HMM()
    hmm.train(train_set)
    known_error_rate, unknown_error_rate, total_error_rate, confusion_matrix = hmm.error_rate(test_set)
    print("Known word error rate:", known_error_rate)
    print("Unknown word error rate:", unknown_error_rate)
    print("Total error rate:", total_error_rate)

    print("D")
    known_error_rate, unknown_error_rate, total_error_rate, confusion_matrix = hmm.error_rate(test_set,
                                                                                              laplace_smooth=1)
    print("Known word error rate:", known_error)
    print("Unknown word error rate:", unknown_error)
    print("Total error rate:", total_error)

    print("E2")
    hmm = HMM()
    hmm.train(train_set, pseudo_words=True)
    known_error_rate, unknown_error_rate, total_error_rate, confusion_matrix = hmm.error_rate(test_set,
                                                                                              pseudo_words=True)
    print("Known word error rate:", known_error_rate)
    print("Unknown word error rate:", unknown_error_rate)
    print("Total error rate:", total_error_rate)

    print("E3")
    known_error_rate, unknown_error_rate, total_error_rate, confusion_matrix = hmm.error_rate(test_set,
                                                                                              laplace_smooth=1,
                                                                                              pseudo_words=True)
    print("Known word error rate:", known_error_rate)
    print("Unknown word error rate:", unknown_error_rate)
    print("Total error rate:", total_error_rate)
    most_significant_error = sorted(confusion_matrix.items(), key=lambda item: item[1], reverse=True)[:15]
    for ((true_tag, predicted_tag), error_num) in most_significant_error:
        print(f"for {true_tag}, viterbi predicted {predicted_tag}, {error_num} times")
