import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

INPUT_DIMENSIONS = [28, 28, 1]
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
LOG_FREQUENCY = 100


def inference(tp_input, reuse=False):
    with tf.variable_scope('mnist_conv', reuse=reuse):
        tv_input_as_image = tf.reshape(tp_input, [-1,INPUT_DIMENSIONS[0],INPUT_DIMENSIONS[1],INPUT_DIMENSIONS[2]])
        te_net = slim.conv2d(tv_input_as_image, 32, [5, 5], reuse=reuse)
        te_net = slim.max_pool2d(te_net, [2, 2])
        te_net = slim.conv2d(te_net, 64, [5, 5], reuse=reuse)
        te_net = slim.max_pool2d(te_net, [2, 2])
        te_net = slim.flatten(te_net)
        te_net = slim.fully_connected(te_net, 1024, reuse=reuse)
        te_net = slim.fully_connected(te_net, 10, activation_fn=None, reuse=reuse)
    return te_net


def loss(te_inference, tp_labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tp_labels, logits=te_inference))


def train(te_loss):
    tv_global_step = tf.Variable(0, name='global_step_d', trainable=False)
    return slim.learning.create_train_op(te_loss, tf.train.AdamOptimizer(learning_rate=LEARNING_RATE), global_step=tv_global_step)


def accuracy(te_inference, tp_labels):
    te_correct_prediction = tf.equal(tf.argmax(te_inference, 1), tf.argmax(tp_labels, 1))
    return tf.reduce_mean(tf.cast(te_correct_prediction, tf.float32))


def placeholders():
    tp_input = tf.placeholder(tf.float32, shape=[None, INPUT_DIMENSIONS[0]*INPUT_DIMENSIONS[1]])
    tp_label = tf.placeholder(tf.float32, shape=[None, 10])
    return tp_input, tp_label


def run_training(nepochs):
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.Graph().as_default():
        # Create placeholder
        tp_input, tp_labels = placeholders()

        # Create network
        te_inference = inference(tp_input)
        te_loss = loss(te_inference, tp_labels)

        # Create train ops
        te_train = train(te_loss)
        te_accuracy = accuracy(te_inference, tp_labels)

        # Create session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # Summaries
        tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tf.summary.scalar('loss', te_loss)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./summaries', sess.graph)

        duration = 0
        print('%-10s | %-20s | %-20s | %-10s' % ('Epoch', 'Loss', 'Accuracy', 'Time(s)'))
        print('-' * 86)

        for i in range(nepochs):

            batch = dataset.train.next_batch(BATCH_SIZE)
            start_time = time.time()
            # Training
            summary, _, val_loss = sess.run([merged, te_train, te_loss], feed_dict={tp_input: batch[0], tp_labels: batch[1],})
            duration += (time.time() - start_time)
            summary_writer.add_summary(summary, i)

            # Logging
            if i % LOG_FREQUENCY == 0:
                print('%-10s | %-20s | %-20s | %-10s' % ('%d' % i, '%.5f' % val_loss, '%.5f' % sess.run(te_accuracy, {tp_input: batch[0], tp_labels: batch[1]}), '%.2f' % duration))
                duration = 0

        # Evaluate Final Test Accuracy
        mnist_test_images = dataset.test.images
        mnist_test_labels = dataset.test.labels
        overall_acc = 0.0
        for i in range(0, len(mnist_test_images), BATCH_SIZE):
            overall_acc += sess.run(te_accuracy, {tp_input: mnist_test_images[i:i + BATCH_SIZE], tp_labels: mnist_test_labels[i:i + BATCH_SIZE]})
        print('Final test accuracy: %g' % (overall_acc * 100 / len(mnist_test_images)))

if __name__=='__main__':
    run_training(2000)