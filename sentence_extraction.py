import numpy as np
import pandas as pd

class SentenceExtraction:
    """Get important sentences from text by TextRank"""

    def __init__(self, sentences_number, column_name):
        self.sentence_tokenized_file = 'csv/comments_revamped_list_20200425.csv'
        self.sentences_origin = pd.read_csv(self.sentence_tokenized_file)[column_name]

        # 센텐스 개수 조절
        self.sentences = self.sentences_origin[1000:2000].tolist()

        self.sentences_size = len(self.sentences)

        # 중요한 문장 number개 추출
        self.number = sentences_number
        self.g = None
        self.important_sentences = None

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

    def get_matrix(self):
        """Make similarity matrix for all sentences"""
        g = np.zeros((self.sentences_size, self.sentences_size), dtype='float')

        for i in range(1, self.sentences_size):  # 1부터 N-1 까지; (i, i) 성분은 0으로 세팅 (?)
            sent1 = self.sentences[i]
            for j in range(i + 1, self.sentences_size):
                sent2 = self.sentences[j]
                g[i][j] = self.similarity(sent1, sent2)
        return g

    def get_most_important(self, g):
        """Get n most important sentences from text"""
        sentences_sum = np.sum(g, axis=1).tolist()
        sentences_sorted = sorted(sentences_sum, reverse=True)[0:self.number]

        important_sentences = list()
        for val in sentences_sorted:
            index = sentences_sum.index(val)
            sentence = self.sentences[index]
            important_sentences.append(sentence)

        return important_sentences

    def analyze(self):
        """Analyze text and extract the most important sentences by measuring similarities"""

        # Get similarity matrix from tokenized sentences
        g = self.get_matrix()
        self.g = g

        # Get the most important N sentences from text
        important_sentences = self.get_most_important(g)
        self.important_sentences = important_sentences

        for sentence in important_sentences:
            print("sentence: ", sentence)
            print("length: ", len(sentence), '\n')


glowpick_analysis = SentenceExtraction(5, 'comments')
glowpick_analysis.analyze()
