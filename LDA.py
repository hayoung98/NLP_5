import jieba
from gensim import corpora, models
import re
import csv


def remove_punctuation(line):
    rule = re.compile(u"[^\u4e00-\u9fa5]")  # 只留下中文字
    line = rule.sub('',line)
    return line

with open('output.csv', newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    with open('stop_word.txt') as file:
        All_sw = file.read()
        stop_word_list = All_sw.splitlines()

    words_list = []  # 分詞
    for row in rows:
        line = (row['Content'])
        line = remove_punctuation(line)
        words = [i for i in jieba.lcut(line, cut_all=False) if i not in stop_word_list]
        words_list.append(words)
    # print(seg_list)
    dictionary = corpora.Dictionary(words_list)
    # print(dictionary)
    corpus = [dictionary.doc2bow(words) for words in words_list]
    # print(corpus)
    for topic_num in range(1,7):
        print("--------------------------------")
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_num)
        for topic in lda.print_topics(num_words=3):
            print(topic)

