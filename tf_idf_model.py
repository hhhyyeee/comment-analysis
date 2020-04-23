import numpy as np
import pandas as pd


# TF-IDF 알고리즘의 특징:
# 문서(댓글) 간의 비슷한 정도를 구함
# 특정 단어가 문서(댓글)에서 얼마나 중요한지 척도를 계산
# 문서(댓글) 내 단어들의 척도를 계산해서 핵심어를 추출
# 검색엔진에서 검색결과의 순위를 결정


class TFIDF:
    """Calculating TF_IDF score for given texts set"""

    def __init__(self):
        self.text = pd.read_csv('csv/comments_revamped_list_20200420.csv')['comments'][2000:2020]

    @staticmethod
    def freq(text, keyword):
        """Return the frequency of keyword in text"""
        return text.count(keyword)

    def tf(self, t, d):
        """문서 d 내에서 단어 t의 증가 빈도 계산"""
        print(d)
        return 0.5 + 0.5 * TFIDF.freq(d, t) / max([TFIDF.freq(d, t) for t in d])

model = TFIDF()