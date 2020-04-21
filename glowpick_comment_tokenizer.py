import re
import csv
from konlpy.tag import Okt


def csvread_list(file_loc):
    raw_list = list()
    with open(file_loc, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            raw_list.append(row[0])
    return raw_list


class GlowpickCommentTokenizer:
    """Tokenize Comments from Glowpick"""

    def __init__(self):
        self.comment_file = 'csv/comments_20200406.csv'
        self.keyword_file = 'csv/tokenize_keywords.csv'
        self.raw_comment_list = csvread_list(self.comment_file)
        self.keyword_list = csvread_list(self.keyword_file)
        self.exclude_classes = ['Josa', 'Exclamation', 'Suffix', 'Determiner', 'Conjunction', 'PreEomi']

    def get_comments_only(self):
        comment_list = list()
        for row in self.raw_comment_list:
            row_1 = row.split("|")
            comment_list.append(row_1[3])
        return comment_list

    def replace_comments_with_keywords(self, comment_list):
        comment_list_replaced = list()
        for sentence in comment_list:
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

        # Get only comments from csv file
        comment_list = self.get_comments_only()

        # Replace words in comments with keywords
        comment_list_replaced = self.replace_comments_with_keywords(comment_list)

        # Tokenize remnants with konlpy Okt tokenizer
        comment_list_tokenized = self.tokenize_comments(comment_list_replaced)

        # Fix comments by re-replacing keywords
        comment_list_fixed = self.fix_comments(comment_list_tokenized)

        return comment_list_fixed


glowpick_comments = GlowpickCommentTokenizer()
glowpick_comments_tokenized = glowpick_comments.tokenize()

print(glowpick_comments_tokenized[0:20])