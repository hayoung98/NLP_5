import jieba
import re
import csv
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct


def remove_punctuation(line):
    rule = re.compile(u"[^\u4e00-\u9fa5]")  # 只留下中文字
    line = rule.sub('',line)
    return line

def Cut(rows, stop_word_list):
    words_list = []  # ex. ['你','我','他'....]
    Label = []  # ex. ['A',...'B',...'C',...]
    for row in rows:
        line = (row['Content'])
        line = remove_punctuation(line)
        words = [i for i in jieba.lcut(line, cut_all=False) if i not in stop_word_list]
        words[:] = ['，'.join(words[:])]
        words_list.append(words[0])
        label = (row['Label'])
        Label.append(label)
    return words_list, Label

if __name__ == '__main__':

    with open('output.csv', newline='') as csvfile:
        with open('stop_word.txt') as file:
            All_sw = file.read()
            stop_word_list = All_sw.splitlines()

        rows = csv.DictReader(csvfile)
        words_list, Label = Cut(rows, stop_word_list)

        vectorizer = CountVectorizer(token_pattern='\\b\\w+\\b')  # 原本只使用2個字以上的詞，改為1個字即可使用
        X = vectorizer.fit_transform(words_list)
        words = vectorizer.get_feature_names()

        topic_model = ct.Corex(n_hidden=8)
        topic_model.fit(X, words=words, docs=Label, anchors=['媒體','經濟','健康'],anchor_strength=2)  # anchors目前是自己設定
        topics = topic_model.get_topics()

        for topic_n,topic in enumerate(topics):
            words, probs = zip(*topic)  # words:關鍵字詞  probs:關鍵字詞的詞頻
            topic_str = str(topic_n+1)+': '+','.join(words[0:6])
            print(topic_str)

        top_docs = topic_model.get_top_docs()
        for topic_n, topic_docs in enumerate(top_docs):
            docs, probs = zip(*topic_docs)
            topic_str = str(topic_n+1)+': '+','.join(docs)
            print(topic_str)
