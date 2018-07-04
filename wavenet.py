import os

import tensorflow as tf
import numpy as np
import argparse
import tqdm
from datasets import SimpleWaveForms
import keras
import coloredlogs, logging
from collections import OrderedDict
from util import next_logdir
import os.path as pth

layers = keras.layers
logger = logging.getLogger("WaveNet")
coloredlogs.install(level="INFO", logger=logger)


class WaveNet:

    def __init__(self, regularizer=tf.keras.regularizers.l2, regularize_coeff=1e-1,
                 dilation_stacks=2, dilation_pow2=5, learning_rate=1e-4, filter_width=2,
                 quantization_channels=256, skip_channels=512, dilation_channels=32,
                 residual_channels=32, global_condition=False, global_cond_depth=3):
        """WaveNet model.
        :param regularizer: Kernel regularizer.
        :param regularize_coeff: Kernel regularizer coefficient.
        :param dilation_stacks: Number of repeated dilation stacks. For example, if
            dilation_pow2 equals 3, and dilation_stacks equals 2, the resulting dilations will be:
            1, 2, 4, 1, 2, 4
        :param dilation_pow2: How many powers of 2 to repeat. E.g. if it equals 4, it will repeat
            dilations of 1, 2, 4 and 8
        :param learning_rate: Learning rate of optimizer
        :param filter_width: Width of causal convolution filters
        :param quantization_channels: Number of quantization channels
        :param skip_channels: Number of channels for skip connections
        :param dilation_channels: Number of channels within dilation block (before going to skip
            or residual)
        :param residual_channels: Number of channels in residual connections
        :param global_condition: Whether to use global conditioning
        :param global_cond_depth: The number of classes for global conditioning
        """
        self.layers = OrderedDict()
        self._regularizer = regularizer(regularize_coeff)
        self._dilations = np.repeat(2 ** np.arange(dilation_pow2), repeats=dilation_stacks)
        self._lr = learning_rate
        self._filter_width = filter_width
        self._quantization_channels = quantization_channels
        self._skip_channels = skip_channels
        self._dilation_channels = dilation_channels
        self._residual_channels = residual_channels
        self._use_global_condition = global_condition
        self._global_cond_depth = global_cond_depth

    def build(self, x, global_condition=None):
        """Builds the network layers for an input x.
        :param x: A Tensor containing one-hot the input of the WaveNet.
        :param global_condition: Global condition input.
        :return: Returns the logits for the quantized output.
        """
        with tf.name_scope("Preprocess"):
            # Preprocess the incoming one-hot quantized representation
            current_out = self._causal_conv(
                x, filters=self._dilation_channels, dilation_rate=1, name="PreprocessConv")

        with tf.name_scope("DilationsStack"):
            skip_connections = []
            for i, dilation_rate in enumerate(self._dilations):
                skip_out, current_out = self.dilation_block(
                    current_out, dilation_rate=dilation_rate, filters=self._dilation_channels,
                    residual_filters=self._residual_channels, skip_filters=self._skip_channels,
                    name="DilationBlock" + str(i), h=global_condition)
                skip_connections.append(skip_out)

        with tf.name_scope("PostProcess"):
            # Post process by combining skip connections followed by two final singular convs
            skip_connections_sum = tf.nn.relu(tf.add_n(skip_connections))
            post_conv0 = tf.nn.relu(
                self._singular_conv(skip_connections_sum, self._skip_channels, name="PostConv0"))
            post_conv1 = self._singular_conv(
                post_conv0, self._quantization_channels, name="PostConv1")

        return post_conv1

    def _causal_conv(self, x, filters, dilation_rate, name="CausalConv"):
        """Builds a causal convolution.
        :param x: The input for the convolution
        :param filters: The number of filters in the convolution
        :param dilation_rate: The dilation rate
        :param name: Name of the layer
        :return: Output of the layer for `x`
        """
        if name not in self.layers:
            self.layers[name] = layers.Conv1D(
                filters=filters, dilation_rate=(dilation_rate,), padding='causal', kernel_size=2,
                kernel_regularizer=self._regularizer, name=pth.basename(name))
        return self.layers[name](x)

    def _global_condition_layer(self, h, filters, name="GlobalContext"):
        """Computes embedding for global condition.
        :param h: Global condition input (a vector for each sample in the batch)
        :param filters: The number of 'filters', i.e. the number of output units.
        :param name: The name of the layer
        :return: Output of the layer for `h`
        """
        if name not in self.layers:
            self.layers[name] = layers.Dense(
                units=filters, kernel_regularizer=self._regularizer, name=pth.basename(name))
        return tf.expand_dims(self.layers[name](h), axis=1)

    def _singular_conv(self, x, filters, name="SingularConv"):
        """Computes a singular convolution, which uses kernel width of 1 (so padding algorithm
        doesn't even matter).
        :param x: The input of the layer
        :param filters: The number of filters
        :param name: The name of the layer
        :return: Output of the layer for `x`
        """
        if name not in self.layers:
            self.layers[name] = layers.Conv1D(
                filters=filters, kernel_size=1, kernel_regularizer=self._regularizer,
                name=pth.basename(name))
        return self.layers[name](x)

    def dilation_block(self, x, dilation_rate, filters, residual_filters, skip_filters,
                       name="DilationBlock", h=None):
        """Computes a dilation block consisting of 2 convolutions for its input sequence `x`. The
        first is used for the 'candidate' output while the second convolution is used as a gate.
        The gated output is then passed on (through convolutions) to a skip output (that skips all
        the way to the layer after the dilation stacks) and a residual output.
        :param x: The input sequence
        :param dilation_rate: The dilation rates for the convolutions within this block
        :param filters: The number of filters for the gated convolution output
        :param residual_filters: The number of filters for the residual output
        :param skip_filters: The number of filters for the skip connection output
        :param name: The name of this block
        :param h: An optional global condition vector (batched of course)
        :return: The output of this dilation block
        """
        with tf.name_scope(name):
            # Normal convolution and gate
            conv_gate_and_filter = self._causal_conv(
                x, filters=filters * 2, dilation_rate=dilation_rate,
                name=name + "/SequenceGateAndFilter")

            if h is not None:
                # Optionally add embedding of global condition
                conv_gate_and_filter += self._global_condition_layer(
                    h, filters=filters * 2, name=name + "/GlobalConditionGateAndFilter")

            # Split the filters and compute gated output
            conv_filter, conv_gate = tf.split(conv_gate_and_filter, num_or_size_splits=2, axis=2)
            gated_act = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)  # Apply gate

            # Compute residual (should have same as number of filters as x) and skip connections
            residual_and_skip = self._singular_conv(
                gated_act, filters=residual_filters + skip_filters, name=name + "/ResidualAndSkip")
            residual_out, skip_out = tf.split(
                residual_and_skip, num_or_size_splits=[residual_filters, skip_filters], axis=2)

            return skip_out, x + residual_out  # Return skip output and apply residuals

    def compile(self, sig_t0, sig_t1, global_cond, optimizer=tf.train.AdamOptimizer, mode='train'):
        """Compiles the graph for a sequence starting at t=0 and its target (which starts at t=1).
        :param sig_t0: Signal starting at t=0
        :param sig_t1: Signal starting at t=1
        :param global_cond: Global condition vector (batched)
        :param optimizer: TensorFlow optimizer
        :param mode: Either 'train' or 'test'
        :return: Ops for: (i) the MLE prediction of the (quantized) signal, (ii) the cross entropy
            loss, the (iii) logits and (iv) the mean squared error.
        """
        logger.info("Compiling for mode '{}'".format(mode.title()))
        with tf.name_scope(mode.title()):

            # Compute one-hot representation of mu_law(x) at t = 0 and mu_law(x) at t = 1
            sig_t0_mu_law = self.mu_law(sig_t0, one_hot=True, name="MuLawT0")
            sig_t1_mu_law = self.mu_law(sig_t1, one_hot=False, name="MuLawT1")

            # Global condition
            if self._use_global_condition:
                with tf.name_scope("GlobalConditionOneHot"):
                    global_cond = tf.one_hot(global_cond, self._global_cond_depth)
            else:
                global_cond = None

            # Compute logits as given by the model
            logits = self.build(sig_t0_mu_law, global_condition=global_cond)
            with tf.name_scope("Predictions"):
                prediction_quantized = tf.argmax(logits, axis=-1)
                prediction_signal = self.mu_law_inverse(prediction_quantized)

            cross_entropy = self.loss(logits=logits, labels=sig_t1_mu_law)  # Compute loss
            with tf.name_scope("Metrics"):
                with tf.name_scope("MSE"):
                    mse = tf.reduce_mean(tf.square(prediction_signal - sig_t1))  # Compute MSE
            if mode == "train":
                with tf.name_scope("Optimize"):
                    minimize = optimizer(learning_rate=self._lr).minimize(cross_entropy)
                return prediction_signal, cross_entropy, minimize, logits, mse
        return prediction_signal, cross_entropy, logits, mse

    def loss(self, logits, labels):
        """Computes mean cross entropy loss (summed within sequence and averaged over
        all sequences).
        :param logits: The output logits as given by the WaveNet
        :param labels: The one-hot labels of the targets
        :return: A Tensor containing the loss
        """
        with tf.name_scope("Loss"):
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            return tf.reduce_mean(tf.reduce_sum(ce, axis=-1))

    def mu_law(self, x, one_hot=True, name="MuLaw"):
        """Computes 'forward' \mu-law where the output is quantized.
        :param x: A batch of scalar sequences.
        :param one_hot: Whether to return the output as onehot representation
        :param name: Name of the computation
        :return: A quantized \mu-law transformation of `x`
        """
        with tf.name_scope(name):
            mu = tf.constant(self._quantization_channels - 1, dtype=tf.float32)
            transformed = tf.sign(x) * tf.nn.softplus(mu * tf.abs(x)) / tf.nn.softplus(mu)
            out = tf.to_int32(tf.round((transformed + 1.0) * mu / 2.0))
            if one_hot:
                return tf.one_hot(out, depth=self._quantization_channels, axis=2)
            return out

    def mu_law_inverse(self, x_one_hot, name="MuLawInverse"):
        """Computes the inverse \mu-law from a one-hot input sequence.
        :param x_one_hot: A one-hot input sequence
        :param name: Name of the computation
        :return: The inverse \mu-law transform of `x_one_hot`
        """
        with tf.name_scope(name):
            x = tf.gather(tf.to_float(
                tf.linspace(-1.0, 1.0, self._quantization_channels)), indices=x_one_hot)
            mu = tf.constant(self._quantization_channels - 1, dtype=tf.float32)
            return tf.sign(x) / mu * (tf.pow(1.0 + mu, tf.abs(x)) - 1.0)


def train(args):
    wavenet = WaveNet(
        regularize_coeff=args.regularize_coeff, learning_rate=args.lr,
        global_condition=args.global_cond, dilation_stacks=args.dilation_stacks,
        filter_width=args.filter_width, quantization_channels=args.quantization_channels,
        dilation_channels=args.dilation_channels, global_cond_depth=args.global_cond_depth,
        residual_channels=args.residual_channels, dilation_pow2=args.dilation_pow2,
        skip_channels=args.skip_channels)

    # Load the data
    logger.info("Loading SimpleWaveForms data")
    dataset = SimpleWaveForms(
        sequence_len=args.sequence_len, freq_range=(args.freq_min, args.freq_max),
        sample_freq=args.sample_freq)

    with tf.name_scope("InputPipeline"):
        (train_t0, train_t1, train_cond), (test_t0, test_t1, test_cond) = dataset.pipeline(
            args.train_size, args.test_size, batch_size=args.batch_size)

    # Compile graph with train data iterator
    train_pred, train_loss, optimize, train_logits, train_mse = wavenet.compile(
        train_t0, train_t1, train_cond, mode='train')

    # Compile graph with test data iterator
    test_pred, test_loss, test_logits, test_mse = wavenet.compile(
        test_t0, test_t1, test_cond, mode='test')

    logger.info("Layer overview")
    for name in wavenet.layers:
        logger.info(name)

    # Setup summaries
    logger.info("Setting up summaries")
    with tf.name_scope("ScalarSummaries"):
        tf.summary.scalar("CrossEntropyLoss", train_loss)
        tf.summary.scalar("MeanSquaredError", train_mse)
        summary_op = tf.summary.merge_all()

    # Do training
    logger.info("Creating session")
    with tf.Session() as sess:

        # Tensorboard writer
        logdir_train, logdir_test = next_logdir()
        logger.info("Logging train results at {}".format(logdir_train))
        logger.info("Logging test  results at {}".format(logdir_test))
        train_writer = tf.summary.FileWriter(logdir=logdir_train, graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir=logdir_test)

        # Initialize
        logger.info("Initializing variables")
        sess.run(tf.global_variables_initializer())

        # Loop lengths
        train_steps = int(np.ceil(args.train_size) / args.batch_size)
        test_steps = int(np.ceil(args.test_size) / args.batch_size)
        for epoch in range(args.num_epochs):

            # Train it
            logger.info("Current epoch: {}".format(str(epoch).zfill(
                int(np.log10(args.num_epochs * 10)))))
            if epoch == 0:
                logger.info("First epoch, graph initialization will take some time.")
            pbar = tqdm.trange(train_steps, desc="Train")
            loss_avg, mse_avg = 0.0, 0.0
            for i in pbar:
                if i % args.summary_interval == 0:
                    loss, _, summary_out, mse = sess.run(
                        [train_loss, optimize, summary_op, train_mse])
                    train_writer.add_summary(
                        summary=summary_out, global_step=epoch * train_steps + i)
                    train_writer.flush()
                else:
                    loss, _, mse = sess.run([train_loss, optimize, train_mse])
                loss_avg = loss_avg + (loss - loss_avg) / (i + 1)
                mse_avg = mse_avg + (mse - mse_avg) / (i + 1)
                pbar.set_postfix(Loss=loss_avg, MSE=mse_avg)

            # Test it
            logger.info("Testing")
            if epoch == 0:
                logger.info("First epoch, graph initialization will take some time.")
            pbar = tqdm.trange(test_steps, desc="Test")
            loss_avg, mse_avg = 0.0, 0.0
            for i in pbar:
                loss, mse = sess.run([test_loss, test_mse])
                loss_avg = loss_avg + (loss - loss_avg) / (i + 1)
                mse_avg = mse_avg + (mse - mse_avg) / (i + 1)
                pbar.set_postfix(Loss=loss_avg, MSE=mse_avg)

            # Add a summary for average test loss and MSE
            loss_summary = tf.Summary.Value(
                tag="ScalarSummaries/CrossEntropyLoss", simple_value=loss_avg)
            mse_summary = tf.Summary.Value(
                tag="ScalarSummaries/MeanSquaredError", simple_value=mse_avg)
            test_writer.add_summary(
                summary=tf.Summary(value=[loss_summary, mse_summary]),
                global_step=(epoch + 1) * train_steps)
            test_writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['sines'], default="sines")
    parser.add_argument("--train_size", default=int(1e3), type=int)
    parser.add_argument("--test_size", default=int(1e3), type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--sample_freq", default=int(16e3), type=int)
    parser.add_argument("--freq_max", default=int(8e3), type=int)
    parser.add_argument("--freq_min", default=int(1e3), type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--sequence_len", default=int(1e4), type=int)
    parser.add_argument("--regularize_coeff", default=1e-1, type=float)
    parser.add_argument("--dilation_pow2", default=8, type=int)
    parser.add_argument("--dilation_stacks", default=4, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--residual_channels", default=32, type=int)
    parser.add_argument("--filter_width", default=2, type=int)
    parser.add_argument("--dilation_channels", default=32, type=int)
    parser.add_argument("--quantization_channels", default=256, type=int)
    parser.add_argument("--skip_channels", default=512, type=int)
    parser.add_argument("--global_cond_depth", default=3, type=int)
    parser.add_argument("--tflogging", action="store_true", dest="tflogging")
    parser.add_argument('--global_cond', action="store_true", dest="global_cond")
    parser.set_defaults(global_cond=False, tflogging=False)
    args = parser.parse_args()

    if not args.tflogging:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    train(args)
