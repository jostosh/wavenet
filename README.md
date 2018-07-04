# wavenet
WaveNet minimal reimplementation

## About this repo
This reimplementation uses `tensorflow` and `keras`. I've added a synthetic dataset which serves as a way to 
quickly demonstrate the training convergence. The data consists of either sine, square or sawtooth waveforms. 
The shape of the waveform can be included as global context by providing a `global_cond` when invoking the training 
script. 

## Code structure
- `wavenet.py`: contains the model and training mechanism
- `datasets.py`: contains a `SimpleWaveForm` dataset that generates sines, sawtooths and square waveforms.
- `util.py`: some non-relevant utilities

## Usage  
The following invocation will at least produce some nice `TensorBoard` visualizations within a few epochs:
```
python train.py --dilation_stacks 2 --dilation_pow2 5 --train_size 1000 --test_size 100 --sequence_len 10000 --global_cond
```
