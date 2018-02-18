"""
Genetic algorithm implementation for training a neural network - doesnt work, probably has some issues?
"""
import time
import threading
import math
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

INPUT_DIMENSIONS = [28, 28, 1]
# Hyper params
BATCH_SIZE = 200
NUM_GENERATIONS = 100
MUTATION_POWER = 0.005
POP_SIZE = 500
NUM_SELECTED_IND = 10
# Display log messages after every LOG_FREQUENCY iterations during training
LOG_FREQUENCY = 1

# Some MLP networks - can be replaced with deeper MLPs or Covnets

def inference_1_layer_mlp(tp_input, reuse=False):
    """
    Construct the neural network with just 1 layer

    :tp_input: input placeholder
    :return: output logits' expression
    """
    with tf.variable_scope('mnist_es', reuse=reuse):
        te_net = slim.fully_connected(tp_input, 10, activation_fn=None, reuse=reuse, scope='layer1')
    return te_net


def inference_2_layer_mlp(tp_input, reuse=False):
    """
    Construct the neural network with just 2 layers

    :tp_input: input placeholder
    :return: output logits' expression
    """
    with tf.variable_scope('mnist_es', reuse=reuse):
        te_net = slim.fully_connected(tp_input, 128, activation_fn=tf.nn.selu, reuse=reuse, scope='layer1')
        te_net = slim.fully_connected(te_net, 10, activation_fn=None, reuse=reuse, scope='layer2')
    return te_net


def reward(te_inference, tp_labels):
    """
    Reward for the current inference, negative of the traditional loss

    :te_inference: expression for logits
    :tp_labels: placeholder for true labels
    :return: reward expression 
    """
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tp_labels, logits=te_inference))


def accuracy(te_inference, tp_labels):
    """
    Construct accuracy expression

    :te_inference: expression for logits
    :tp_labels: true label placeholder
    :return: accuracy expression
    """
    te_correct_prediction = tf.equal(tf.argmax(te_inference, 1), tf.argmax(tp_labels, 1))
    return tf.reduce_mean(tf.cast(te_correct_prediction, tf.float32))


def placeholders():
    """
    Creates placeholders for inputs and labels
    """ 
    tp_input = tf.placeholder(tf.float32, shape=[None, INPUT_DIMENSIONS[0] * INPUT_DIMENSIONS[1]])
    tp_label = tf.placeholder(tf.float32, shape=[None, 10])
    return tp_input, tp_label


def iterate_minibatches(input_set, target_set, batch_size, shuffle=False):
    """
    Generator to yield minibatches for a training set

    :input_set: input feature set
    :target_set: target labels for the features
    :batch_size: batch size for minibatch
    :shuffle: shuffle the data
    """
    if shuffle:
        indices = np.arange(len(input_set))
        np.random.shuffle(indices)
    for start_idx in range(0, len(input_set) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield input_set[excerpt], target_set[excerpt]


def train(num_generations, mutation_power, pop_size, num_selected_ind, resume=False):
    """
    Train the neural network using a simple genetic algorithm

    :num_generations: number of generations in GA
    :mutation_power: std dev of perturbations
    :pop_size: number of members in each population
    :num_selected_ind: number of selected individuals from the population
    :resume: load a saved model and resume from there
    """
    def normalize_weights(w):
        """
        Normalize weights - a good initialization for GAs
        """
        w *= 1.0 / np.sqrt(np.square(w).sum(axis=0, keepdims=True))
        return w

    def create_feed_dict(x, t, params):
        """
        Utility for creating feed dictionary
        """
        f_dict = {tp_input: x, tp_labels: t}
        for te_l_p, param in zip(te_layer_params, params):
            f_dict[te_l_p] = param
        return f_dict
    
    # load dataset
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    # for now just work on subset of original dataset
    train_images = dataset.train.images[:2000]
    train_labels = dataset.train.labels[:2000]

    with tf.Graph().as_default():
        # create the network and reward, accuracy expressions
        tp_input, tp_labels = placeholders()
        te_inference = inference_2_layer_mlp(tp_input)
        te_reward = reward(te_inference, tp_labels)
        te_accuracy = accuracy(te_inference, tp_labels)

        # initialize all parameters
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # logging
        summary_writer = tf.summary.FileWriter('./summaries_ga', sess.graph)
        saver = tf.train.Saver()

        # create initial param vector
        te_layer_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mnist_ga')
        params = [[] for _ in range(pop_size)]
        fitness = np.zeros((pop_size,))
        if resume:
            params = np.load("./saved_models/mnist_ga_params.npy")
        else:
            for i in range(pop_size):
                # initialize the individual
                for te_p in te_layer_params:
                    # for weights do initialization with normal distribution
                    if "biases" not in te_p.name:
                        params[i].append(normalize_weights(sess.run(tf.random_normal(te_p.shape, stddev=mutation_power))))
                    # for biases, initialize with zeros
                    else:
                        params[i].append(sess.run(te_p))
                # evaluate it's fitness
                for batch in iterate_minibatches(train_images, train_labels, BATCH_SIZE, shuffle=True):
                    fitness[i] += sess.run(te_reward, feed_dict=create_feed_dict(batch[0], batch[1], params[i]))

        # simple genetic algorithm
        for g in range(num_generations):
            # sort params by fitness
            param_index_sorted = [x for (y, x) in sorted(zip(fitness, range(pop_size)), key=lambda pair: pair[0], reverse=True)]
            # initialize next gen params and fitness as 0
            next_gen_params = [[] for _ in range(pop_size + 1)]
            next_gen_fitness = np.zeros((pop_size + 1,))
            # include elite of previous generation as 1st member of new population
            next_gen_params[0] = params[param_index_sorted[0]]
            next_gen_fitness[0] = fitness[param_index_sorted[0]]
            # do logging
            if g % LOG_FREQUENCY == 0:
                print(fitness.shape)
                summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="F-0", simple_value=fitness[param_index_sorted[0]]),
                ]), g)
                summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="F-1", simple_value=fitness[param_index_sorted[1]]),
                ]), g)
                summary_writer.add_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="F-2", simple_value=fitness[param_index_sorted[2]]),
                ]), g)
            # for each member of new pop, select a new member as the perturbed variant of top "num_select_ind" members of previous pop
            for i in range(pop_size):
                selected_index = param_index_sorted[random.randint(0, num_selected_ind - 1)]
                next_gen_params[i + 1] = params[selected_index]
                for next_gen_param_idx in range(len(next_gen_params[i + 1])):
                    next_gen_params[i + 1][next_gen_param_idx] = next_gen_params[i + 1][next_gen_param_idx] + mutation_power * np.random.normal(0, 1, next_gen_params[i + 1][next_gen_param_idx].shape)
                for batch in iterate_minibatches(train_images, train_labels, BATCH_SIZE, shuffle=True):
                    next_gen_fitness[i + 1] += sess.run(te_reward, feed_dict=create_feed_dict(batch[0], batch[1], next_gen_params[i + 1]))
            # set next iterations params and fitness
            params = next_gen_params
            fitness = next_gen_fitness


if __name__ == '__main__':
    train(NUM_GENERATIONS, MUTATION_POWER, POP_SIZE, NUM_SELECTED_IND)
