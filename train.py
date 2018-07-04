import argparse

import numpy as np
import tensorflow as tf
import tqdm

from datasets import SimpleWaveForms
from util import next_logdir
from wavenet import WaveNet, logger
import os


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
