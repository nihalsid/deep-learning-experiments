import time
import threading
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

INPUT_DIMENSIONS = [28, 28, 1]
BATCH_SIZE = 200
LOG_FREQUENCY = 1


def inference_1_layer_mlp(tp_input, reuse=False):
    with tf.variable_scope('mnist_es', reuse=reuse):
        te_net = slim.fully_connected(tp_input, 10, activation_fn=None, reuse=reuse, scope='layer1')
    return te_net


def inference_2_layer_mlp(tp_input, reuse=False):
    with tf.variable_scope('mnist_es', reuse=reuse):
        te_net = slim.fully_connected(tp_input, 128, activation_fn=tf.nn.selu, reuse=reuse, scope='layer1')
        te_net = slim.fully_connected(te_net, 10, activation_fn=None, reuse=reuse, scope='layer2')
    return te_net


def reward(te_inference, tp_labels):
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tp_labels, logits=te_inference))


def accuracy(te_inference, tp_labels):
    te_correct_prediction = tf.equal(tf.argmax(te_inference, 1), tf.argmax(tp_labels, 1))
    return tf.reduce_mean(tf.cast(te_correct_prediction, tf.float32))


def placeholders():
    tp_input = tf.placeholder(tf.float32, shape=[None, INPUT_DIMENSIONS[0] * INPUT_DIMENSIONS[1]])
    tp_label = tf.placeholder(tf.float32, shape=[None, 10])
    return tp_input, tp_label


def iterate_minibatches(input_set, target_set, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(input_set))
        np.random.shuffle(indices)
    for start_idx in range(0, len(input_set) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield input_set[excerpt], target_set[excerpt]

#0.005, 0.01
def train(n_epochs, population_size=50, learning_rate=0.001, sigma=0.01, n_workers=4, resume=False):
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_images = dataset.train.images#[:600]
    train_labels = dataset.train.labels#[:600]

    def fitness_shaping(rewards):
        """
        A rank transformation on the rewards, which reduces the chances
        of falling into local optima early in training.

        Borrowed from https://github.com/atgambardella/pytorch-es/blob/master/train.py#L86
        """
        sorted_rewards_backwards = sorted(rewards)[::-1]
        lamb = len(rewards)
        shaped_rewards = []
        denom = sum([max(0, math.log(lamb / 2 + 1, 2) - math.log(sorted_rewards_backwards.index(r) + 1, 2)) for r in rewards])
        for r in rewards:
            num = max(0, math.log(lamb / 2 + 1, 2) - math.log(sorted_rewards_backwards.index(r) + 1, 2))
            shaped_rewards.append(num / denom + 1 / lamb)
        return shaped_rewards

    def create_feed_dict(x, t, params):
        f_dict = {tp_input: x, tp_labels: t}
        for te_l_p, param in zip(te_layer_params, params):
            f_dict[te_l_p] = param
        return f_dict

    def worker(i, perturbed_params, rewards):
        for batch in iterate_minibatches(train_images, train_labels, BATCH_SIZE, shuffle=True):
            rewards[i] += sess.run(te_reward, feed_dict=create_feed_dict(batch[0], batch[1], perturbed_params))

    with tf.Graph().as_default():

        # create network and reward/accuracy expressions
        tp_input, tp_labels = placeholders()
        te_inference = inference_2_layer_mlp(tp_input)
        te_reward = reward(te_inference, tp_labels)
        te_accuracy = accuracy(te_inference, tp_labels)

        # create session
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)
        sess.run(init)

        # logging
        duration = 0
        print('%-10s | %-20s | %-20s | %-10s' % ('Step', 'Reward', 'Accuracy', 'Time(s)'))
        print('-' * 86)
        summary_writer = tf.summary.FileWriter('./summaries_es', sess.graph)
        saver = tf.train.Saver()

        # create initial param vector
        te_layer_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mnist_es')
        params = []
        if resume:
            params = np.load("./saved_models/mnist_es_params.npy")
        else:
            for te_p in te_layer_params:
                params.append(sess.run(te_p))

        # train for specified number of epochs
        for epoch in range(n_epochs):
            start_time = time.time()

            seeds = []
            perturbed_params = []
            rewards = [0] * population_size

            for _ in range(int(population_size / 2)):
                np.random.seed()
                seeds.append(np.random.randint(2 ** 30))
                seeds.append(seeds[-1])
                perturbed_params.append([])
                perturbed_params.append([])
                np.random.seed(seeds[-1])
                for param in params:
                    perturbed_params[-2].append(param + sigma * np.random.normal(0, 1, param.shape))
                    perturbed_params[-1].append(param - sigma * np.random.normal(0, 1, param.shape))

            for worker_batch_idx in range(int(population_size / n_workers)):
                processes = []
                for worker_idx in range(n_workers):
                    i = worker_batch_idx * n_workers + worker_idx
                    p = threading.Thread(target=worker, args=(i, perturbed_params[i], rewards))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()

            # logging
            val_reward = np.mean(rewards)
            summary_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="reward", simple_value=val_reward),
            ]), epoch)
            # fitness shaping
            shaped_rewards = fitness_shaping(rewards)

            # parameter update
            sign = 1
            for pop_idx in range(int(population_size)):
                np.random.seed(seeds[pop_idx])
                for i in range(len(params)):
                    params[i] = params[i] + sign * learning_rate / (population_size * sigma) * shaped_rewards[pop_idx] * np.random.normal(0, 1, params[i].shape)
                sign *= -1

            duration += (time.time() - start_time)
            # logging
            if epoch % LOG_FREQUENCY == 0:
                for batch in iterate_minibatches(train_images, train_labels, BATCH_SIZE, shuffle=True):
                    val_acc = sess.run(te_accuracy, feed_dict=create_feed_dict(batch[0], batch[1], params))
                    print('%-10s | %-20s | %-20s | %-10s' % ('%d' % (epoch), '%.5f' % val_reward, '%.5f' % val_acc, '%.2f' % duration))
                    summary_writer.add_summary(tf.Summary(value=[
                        tf.Summary.Value(tag="accuracy", simple_value=val_acc),
                    ]), epoch)
                    break
                duration = 0

        # evaluate Final Test Accuracy
        mnist_test_images = dataset.test.images
        mnist_test_labels = dataset.test.labels
        np.save("./saved_models/mnist_es_params.npy", np.array(params))
        overall_acc = 0.0
        for i in range(0, len(mnist_test_images), BATCH_SIZE):
            overall_acc += sess.run(te_accuracy, feed_dict=create_feed_dict(mnist_test_images[i:i + BATCH_SIZE], mnist_test_labels[i:i + BATCH_SIZE], params))
        print('\nFinal test accuracy: %g' % (overall_acc * BATCH_SIZE / len(mnist_test_images)))


if __name__ == '__main__':
    train(1500, population_size=40, n_workers=2, resume=True)
