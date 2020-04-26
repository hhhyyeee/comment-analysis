import pandas as pd
import re
import csv
from konlpy.tag import Okt


class GlowpickCommentTokenizer:
    """Tokenize Comments from Glowpick"""

    @staticmethod
    def test_tokenize(sentence):
        return print(Okt().pos(sentence))

    def __init__(self, comment_list, keyword_list):
        self.comment_list = comment_list
        self.keyword_list = keyword_list
        self.exclude_classes = ['Josa', 'Exclamation', 'Suffix', 'Determiner', 'Conjunction', 'PreEomi']

    def replace_comments_with_keywords(self):
        comment_list_replaced = list()
        for sentence in self.comment_list:
            sentence = ' '.join(re.compile('[가-힣]+').findall(sentence))
            for word in self.keyword_list:
                sentence = sentence.replace(word, str(self.keyword_list.index(word)))
            comment_list_replaced.append(sentence)
        return comment_list_replaced

    def tokenize_comments(self, comment_list_replaced):
        comment_list_tokenized = list()
        okt = Okt()
        for sentence in comment_list_replaced:
            pos = okt.pos(sentence)
            comment_list_tokenized.append(pos)
        return comment_list_tokenized

    def fix_comments(self, comment_list_tokenized):
        comment_list_fixed = list()
        for comment in comment_list_tokenized:
            comm = list()
            for token in comment:
                if token[1] == 'Number':
                    num_comp = re.findall(r'\d+', token[0])
                    num = int(num_comp[0])
                    try:
                        comm.append(self.keyword_list[num])
                    except:
                        print(token[0], ', ', token[1])
                else:
                    if token[1] not in self.exclude_classes:
                        comm.append(token[0])
            fixed_comment = ' '.join(comm)
            comment_list_fixed.append(fixed_comment)
        return comment_list_fixed

    def tokenize(self):
        """Main function to tokenize multiple comments"""

        # Replace words in comments with keywords
        comment_list_replaced = self.replace_comments_with_keywords()

        # Tokenize remnants with konlpy Okt tokenizer
        comment_list_tokenized = self.tokenize_comments(comment_list_replaced)

        # Fix comments by re-replacing keywords
        comment_list_fixed = self.fix_comments(comment_list_tokenized)

        # Convert the string into DataFrame, and export it as a csv file
        df = pd.DataFrame(comment_list_fixed, columns=['comments'])
        df.to_csv("csv/comments_revamped_list_20200426.csv", index=False)

        return comment_list_fixed


comments = list()
keywords = list()

with open('csv/comments_20200406.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    for row in reader:
        comments.append(row[3])

with open('csv/tokenize_keywords.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        keywords.append(row[0])


model = GlowpickCommentTokenizer(comments, keywords)
model.tokenize()