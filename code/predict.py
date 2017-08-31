import tensorflow as tf
from processing_data import *
import numpy as np

word_dimension = 300
batch_size = 128
num_classes = 2
interations = 100000

all_words = get_all_words()
model = gensim.models.Word2Vec.load('baby.model')

def get_words_matrix():
    arr = np.zeros([len(all_words), word_dimension], dtype='float32')
    for i in range(len(all_words)):
        arr[i] = model[all_words[i]]
    return arr


tf.reset_default_graph()
input_data = tf.placeholder(tf.float32, [len(all_words), word_dimension])
weight = tf.Variable(tf.truncated_normal([word_dimension, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
prediction = tf.nn.softmax(tf.matmul(input_data, weight) + bias)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

input_words = get_words_matrix()
prediction_result = sess.run(prediction, {input_data: input_words})
print(list(prediction_result[i] for i in range(10)))
# predict_keywords = open('predict_keywords.txt', 'w')
#
# for i in range(len(prediction_result)):
#     if prediction_result[i][0] > prediction_result[i][1]:
#         predict_keywords.write(all_words[i] + '\n')

# print(sess.run(weight))
# print(sess.run(bias))

