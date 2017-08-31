import jieba
import csv
import gensim


def word_segment(f_name1, f_name2):
    comments_cut = open(f_name2, 'w')
    lines = open(f_name1).readlines()
    cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), lines))
    for line in cut_lines:
        comments_cut.write(line + '\n')
    comments_cut.close()


def get_keywords(f_name, col_id):
    words_list = []
    with open(f_name) as f:
        f_csv = csv.reader(f)
        headding = next(f_csv)
        for row in f_csv:
            words_list.append(row[col_id])
    return words_list

def get_all_words():
    words_list= []
    w2v = gensim.models.Word2Vec.load('baby.model')
    for key, value in w2v.wv.vocab.items():
        words_list.append(key)
    return words_list

def get_pure_predict_words():
    keywords = get_keywords('keyword.csv', 2)
    predict_words = []
    lines = open('predict_keywords.txt').readlines()
    for line in lines:
        predict_words.append(line.replace('\n', ''))
    predict_words = list(set(predict_words) - set(keywords))
    pure_predict_words = open('pure_predict_words', 'w')
    for word in predict_words:
        pure_predict_words.write(word + '\n')



if __name__ == '__main__':
    # word_segment('comments.txt', 'comments_cut.txt')
    #get_pure_predict_words()
    all_words = get_all_words()
    keywords = get_keywords('keyword.csv', 2)
    negtive_word = list(set(all_words) - set(keywords))
    print(len(all_words) - len(negtive_word))