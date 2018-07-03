import tensorflow as tf
import numpy as np
import scipy.signal as signal


class SimpleWaveForms:

    def __init__(self, sequence_len=int(1e4), freq_range=(85, 8e3), sample_freq=16e3,
                 noise_level=1e-1):
        self._sequence_len = sequence_len
        self._freq_min, self._freq_max = freq_range
        self._sample_freq = sample_freq
        self._noise_level = noise_level

    def pipeline(self, train_size, test_size, batch_size=1):
        train_ds = tf.data.Dataset.from_tensor_slices(self._generate(train_size + 1))\
            .cache()\
            .shuffle(buffer_size=train_size)\
            .batch(batch_size)\
            .repeat()\
            .make_one_shot_iterator()

        test_ds = tf.data.Dataset.from_tensor_slices(self._generate(test_size + 1))\
            .cache()\
            .batch(batch_size)\
            .repeat()\
            .make_one_shot_iterator()

        train_t0, train_t1, train_labels = train_ds.get_next()
        test_t0, test_t1, test_labels = test_ds.get_next()

        return (train_t0, train_t1, train_labels), (test_t0, test_t1, test_labels)

    def _generate(self, n):
        num_sines = n // 3
        num_squares = n // 3
        num_sawtooths = n - num_sines - num_squares

        # Set time and frequency
        t = np.divide(np.arange(0, self._sequence_len), self._sample_freq).reshape(
            (1, self._sequence_len))

        def _random_freqs(n):
            return np.random.rand(n).reshape((n, 1)) * (
                    self._freq_max - self._freq_min) + self._freq_min

        # Generate waveforms
        sawtooths = signal.sawtooth(2 * np.pi * _random_freqs(num_sawtooths) * t)
        squares = signal.square(2 * np.pi * _random_freqs(num_squares) * t)
        sines = np.cos(2 * np.pi * _random_freqs(num_sines) * t)

        # Concatenate and add noise
        sig = np.concatenate([sawtooths, sines, squares], axis=0) + np.random.normal(
            scale=self._noise_level, size=n * self._sequence_len).reshape((n, self._sequence_len))

        # Clip and split
        sig = np.clip(sig, -1.0, 1.0).astype(np.float32)

        # Waveform labels
        waveform_labels = np.concatenate([
            np.zeros(num_sawtooths), np.ones(num_sines), np.ones(num_squares) * 2]).astype(np.int32)
        return sig[:, :-1], sig[:, 1:], waveform_labels
