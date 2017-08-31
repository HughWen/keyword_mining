import numpy as np
from processing_data import *
from random import randint
import tensorflow as tf
import datetime

word_dimension = 300
batch_size = 128
num_classes = 2
interations = 100000


all_words = get_all_words()
keywords = get_keywords('keyword.csv', 2)
negtive_word = list(set(all_words) - set(keywords))
keywords = list(set(keywords) & set(all_words))
model = gensim.models.Word2Vec.load('baby.model')


def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, word_dimension], dtype='float32')
    for i in range(batch_size):
        if i % 2 == 0:
            num = randint(0, len(keywords) - 1)
            arr[i] = model[keywords[num]]
            labels.append([1, 0])
        else:
            num = randint(0, len(negtive_word) - 1)
            arr[i] = model[negtive_word[num]]
            labels.append([0, 1])
    return arr, labels


def train():
    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    input_data = tf.placeholder(tf.float32, [batch_size, word_dimension])

    weight = tf.Variable(tf.truncated_normal([word_dimension, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    prediction = (tf.matmul(input_data, weight) + bias)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.InteractiveSession()
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = 'tensorboard/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
    writer = tf.summary.FileWriter(logdir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(interations):
        next_batch, next_batch_labels = get_train_batch()
        sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels})
        if i % 50 == 0:
            summary = sess.run(merged, {input_data: next_batch, labels: next_batch_labels})
            writer.add_summary(summary, i)
            print('step %s' %i)

        if i % 10000 == 0 and i != 0:
            save_path = saver.save(sess, 'models/pretrain_lstm.ckpt', global_step=i)
            print('saved to %s' % save_path)
    writer.close()

if __name__ == '__main__':
    train()
