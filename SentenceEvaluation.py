import numpy as np
import pandas as pd
import csv
import random


class SentenceEvaluation:
    """"""

    @staticmethod
    def similarity(sent1, sent2):
        """Calculate similarity between two sentences"""
        similarity = 0
        sent1_tokens = sent1.split()
        sent2_tokens = sent2.split()
        sent_abs = np.log(len(sent1_tokens)) + np.log(len(sent2_tokens))

        for token in sent1_tokens:
            if token in sent2_tokens:
                similarity = similarity + 1 / sent_abs

        return similarity

    def __init__(self, weights, test_sentences, sentence_size=50):
        self.weights = weights
        self.test_sentences = test_sentences
        self.sentence_size = sentence_size

    def make_ideal_sentences(self):
        words = [weight[0] for weight in weights[:200]]
        longwords = [weight[0] for weight in weights[:200] if len(weight[0]) != 1]

        ideals = list()
        for i in range(5):  # ideal sentence 는 몇 개나 필요한지
            ideals.append(' '.join(random.sample(words, self.sentence_size)))

        return ideals

    def get_scores(self, ideals):
        scores = list()
        for comment in self.test_sentences:
            sim = list()
            for ideal in ideals:
                sim.append(SentenceEvaluation.similarity(comment, ideal))
            mean = np.mean(sim)
            scores.append([comment, mean])
        return scores

    def evaluate(self):
        ideals = self.make_ideal_sentences()
        scores = self.get_scores(ideals)
        return scores


weights = list()
with open('csv/token_weight_20200426.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        weights.append(row)

test_comments = list()
with open('csv/eval_test_tokenized.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_comments.append(row[0])

model = SentenceEvaluation(weights, test_comments)
print(model.evaluate())