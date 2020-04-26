import pandas as pd
from math import log10


# TF-IDF 알고리즘의 특징:
# 문서(댓글) 간의 비슷한 정도를 구함
# 특정 단어가 문서(댓글)에서 얼마나 중요한지 척도를 계산
# 문서(댓글) 내 단어들의 척도를 계산해서 핵심어를 추출
# 검색엔진에서 검색결과의 순위를 결정


class TFIDF:
    """Calculating TF_IDF score for given text"""

    def __init__(self, text):
        self.text = text

    def freq(self, t):
        """Return the frequency of word in text"""
        count = 0
        for sentence in self.text:
            count += sentence.count(t)
        return count

    def max_freq(self):
        max_freq = ['', 0]
        for sentence in self.text:
            for token in sentence.split():
                freq = [token, self.freq(token)]
                if freq[1] > max_freq[1]:
                    max_freq = freq
        return max_freq

    def tf_aug(self, t):
        """Calculate Augmented Frequency (Double Normalization 0.5) of word t in text d"""
        return 0.5 + 0.5 * self.freq(t) / self.max_freq()

    def idf(self, t):
        """Calculate Inverse Document Frequency"""
        text_len = len(self.text)
        denominator = 1 + len([True for d in self.text if t in d])
        return log10(text_len / denominator)

    def tfidf(self, t, d):
        return TFIDF.tf_aug(t) * self.idf(t)

    def tfidfScorer(self, d):
        tokens = d.split()
        scores = list()
        for token in tokens:
            scores.append([token, self.tfidf(token, d)])
        return scores


text = pd.read_csv('csv/comments_revamped_list_20200420.csv')