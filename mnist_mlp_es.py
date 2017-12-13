import time
import threading
from collections import deque
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

INPUT_DIMENSIONS = [28, 28, 1]
BATCH_SIZE = 200
LOG_FREQUENCY = 100


def inference(tp_input, reuse=False):
    with tf.variable_scope('mnist_conv', reuse=reuse):
        te_net = slim.fully_connected(tp_input, 10, activation_fn=None, reuse=reuse, scope='layer1')
    return te_net


def reward(te_inference, tp_labels):
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tp_labels, logits=te_inference))


def accuracy(te_inference, tp_labels):
    te_correct_prediction = tf.equal(tf.argmax(te_inference, 1), tf.argmax(tp_labels, 1))
    return tf.reduce_mean(tf.cast(te_correct_prediction, tf.float32))


def placeholders():
    tp_input = tf.placeholder(tf.float32, shape=[None, INPUT_DIMENSIONS[0]*INPUT_DIMENSIONS[1]])
    tp_label = tf.placeholder(tf.float32, shape=[None, 10])
    return tp_input, tp_label


def train(n_epochs, population_size=50, learning_rate=0.001, sigma=0.1, n_workers=4):

    def flatten_params(weights, biases):
        return np.append(weights.flatten(),biases.flatten())

    def unflatten_params(params, weights_shape, biases_shape):
        weights_length = weights_shape[0] * weights_shape[1]
        weights = np.reshape(params[:weights_length], weights_shape)
        biases = params[weights_length:]
        return weights, biases

    def worker(i, reward, te_reward, batch, noise, params, w_shape, b_shape):
        p = params + sigma * noise[i]
        trial_w_0, trial_b_0 = unflatten_params(p, w_shape, b_shape)
        reward[i] = sess.run(te_reward, feed_dict={tp_input: batch[i][0], tp_labels: batch[i][1], te_w_0: trial_w_0, te_b_0: trial_b_0})

    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Graph().as_default():
        tp_input, tp_labels = placeholders()
        te_inference = inference(tp_input)
        te_reward = reward(te_inference, tp_labels)
        te_accuracy = accuracy(te_inference, tp_labels)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)
        sess.run(init)

        duration = 0
        print('%-10s | %-20s | %-20s | %-10s' % ('Epoch', 'Reward', 'Accuracy', 'Time(s)'))
        print('-' * 86)
        summary_writer = tf.summary.FileWriter('./summaries_es', sess.graph)
        saver = tf.train.Saver()

        te_layer_params_0 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mnist_conv/layer1')
        te_w_0 = te_layer_params_0[0]
        te_b_0 = te_layer_params_0[1]
        w_0 = sess.run(te_w_0)
        b_0 = sess.run(te_b_0)
        params_0 = flatten_params(w_0, b_0)
        for epoch in range(n_epochs):
            start_time = time.time()

            noise_0 = np.random.randn(population_size, params_0.shape[0])
            rewards_0 = np.zeros(population_size)

            batches = []

            for i1 in range(int(population_size/n_workers)):
                jobs = []
                for i2 in range(n_workers):
                    batches.append(dataset.train.next_batch(BATCH_SIZE))
                    job = threading.Thread(target=worker, args=(i1*n_workers+i2, rewards_0, te_reward, batches, noise_0, params_0, w_0.shape, b_0.shape))
                    jobs.append(job)
                    job.start()
                for job in jobs:
                    job.join()

            normalized_rewards_0 = (rewards_0 - np.mean(rewards_0)) / np.std(rewards_0)
            params_0 = params_0 + learning_rate / (population_size * sigma) * np.dot(noise_0.T, normalized_rewards_0)
            duration += (time.time() - start_time)

            summary_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="reward", simple_value=np.mean(rewards_0)),
            ]), epoch)

            if epoch % LOG_FREQUENCY == 0:
                batch = dataset.train.next_batch(BATCH_SIZE)
                w_0, b_0 = unflatten_params(params_0, w_0.shape, b_0.shape)
                val_acc = sess.run(te_accuracy, {tp_input: batch[0], tp_labels: batch[1], te_w_0: w_0, te_b_0: b_0})
                print('%-10s | %-20s | %-20s | %-10s' % ('%d' % epoch, '%.5f' % np.mean(rewards_0), '%.5f' % val_acc, '%.2f' % duration))
                summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="accuracy", simple_value=val_acc),
                ]), epoch)
                duration = 0

        # Evaluate Final Test Accuracy
        mnist_test_images = dataset.test.images
        mnist_test_labels = dataset.test.labels
        w_0, b_0 = unflatten_params(params_0, w_0.shape, b_0.shape)
        saver.save(sess, "./saved_models/mnist_es.ckpt")
        np.save("./saved_models/mnist_es_w_0.npy", w_0)
        np.save("./saved_models/mnist_es_b_0.npy", b_0)
        overall_acc = 0.0
        for i in range(0, len(mnist_test_images), BATCH_SIZE):
            overall_acc += sess.run(te_accuracy, {tp_input: mnist_test_images[i:i + BATCH_SIZE], tp_labels: mnist_test_labels[i:i + BATCH_SIZE], te_w_0: w_0, te_b_0: b_0})
        print('\nFinal test accuracy: %g' % (overall_acc * BATCH_SIZE / len(mnist_test_images)))


def test():
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Graph().as_default():
        tp_input, tp_labels = placeholders()
        te_inference = inference(tp_input)
        te_accuracy = accuracy(te_inference, tp_labels)
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)
        sess.run(init)
        te_layer_params_0 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mnist_conv/layer1')
        te_w_0 = te_layer_params_0[0]
        te_b_0 = te_layer_params_0[1]

        mnist_test_images = dataset.train.images
        mnist_test_labels = dataset.train.labels

        w_0 = np.load("./saved_models/mnist_es_w_0.npy")
        b_0 = np.load("./saved_models/mnist_es_b_0.npy")

        overall_acc = 0.0
        for i in range(0, len(mnist_test_images), BATCH_SIZE):
            overall_acc += sess.run(te_accuracy, {tp_input: mnist_test_images[i:i + BATCH_SIZE], tp_labels: mnist_test_labels[i:i + BATCH_SIZE], te_w_0: w_0, te_b_0: b_0})
        print('\nFinal test accuracy: %g' % (overall_acc * BATCH_SIZE / len(mnist_test_images)))

if __name__ == '__main__':
    #train(10000, population_size=50, n_workers=5)
    test()
