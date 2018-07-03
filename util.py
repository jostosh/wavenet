import os
import os.path as pth


def next_logdir(base="./logs"):
    os.makedirs(base, exist_ok=True)
    subdirs = [d for d in os.listdir(base) if pth.isdir(pth.join(base, d))]
    logdir = pth.join(base, "run" + str(len(subdirs)).zfill(4))
    logdir_train = pth.join(logdir, "train")
    logdir_test = pth.join(logdir, "test")
    return logdir_train, logdir_test
