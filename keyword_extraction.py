import numpy as np
import pandas as pd
from collections import OrderedDict


def symmetrize(a):
    """Get a square matrix and return its decalcomanie"""
    return a + a.T - np.diag(a.diagonal())


class KeywordExtraction:
    """Main function to extract keywords"""

    def __init__(self):
        self.window_size = 7
        self.d = 0.85
        self.min_diff = 1e-5
        self.steps = 10
        self.node_weight = None
        self.comment_tokenized_file = 'csv/comments_revamped_list_20200420.csv'

    def prepare(self):
        """Get tokenized comments from csv file, make two lists"""
        # [comment0, comment1, ...]
        # [[token00, token01, ...], [token10, token11, ...], ...]

        tokens_comments_list = list()
        df_comments_list = pd.read_csv(self.comment_tokenized_file)
        comments_list = df_comments_list['comments'].tolist()
        for comment in comments_list:
            tokens_comments_list.append(comment.split())
        return comments_list, tokens_comments_list

    def get_token_pairs(self, comment):
        """Build token pairs per sentence with certain window size"""
        token_pairs = list()
        for i, word in enumerate(comment):
            for j in range(i + 1, i + self.window_size):
                if j >= len(comment):
                    break
                pair = (word, comment[j])
                if pair not in token_pairs:
                    token_pairs.append(pair)
        return token_pairs

    def get_vocab(self, tokens_pairs_list):
        """Get all tokens from comments list"""
        vocab = OrderedDict()
        i = 0
        for comment in tokens_pairs_list:
            for token in comment:
                if token not in vocab:
                    vocab[token] = i
                    i += 1
        return vocab

    def get_matrix(self, vocab, token_pairs_list):
        """Get transition matrix for vocab"""
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs_list:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        g = symmetrize(g)

        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=(norm != 0))
        return g_norm

    def analyze(self):
        """Analyze comments by token frequencies, get normalized matrix for Markov chain"""

        # Get comments, tokens from csv
        comments_list, tokens_comments_list = self.prepare()

        # Get token pairs of comments
        token_pairs_list = list()
        for comment in comments_list:
            token_pairs_list.append(self.get_token_pairs(comment))

        # Get all tokens
        vocab = self.get_vocab(token_pairs_list)

        # Get transition matrix
        g_norm = self.get_matrix(vocab, token_pairs_list)

        # Initialization for weight(PageRank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g_norm, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight
