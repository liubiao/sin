import cPickle
import os

import numpy as np
import theano

from const import *


def prepare_data(dx, dy):
    s_maxlen = max([max([len(s) for s in d]) for d in dx])
    s_num = sum([len(d) for d in dx])
    d_samples = len(dx)

    x = np.zeros((s_num + d_samples - 1, s_maxlen)).astype('int32')
    m = np.zeros((s_num + d_samples - 1, s_maxlen)).astype(theano.config.floatX)
    y = np.zeros((s_num + d_samples - 1, )).astype('int32')

    idx = 0
    for td, ty in zip(dx, dy):
        y[idx:idx+len(ty)] = ty
        for s in td:
            s_len = len(s)
            x[idx, :s_len] = s
            m[idx, :s_len] = 1
            idx += 1
        # use a blank sentence to separate dialogues
        idx += 1
    return x, m, y


def load_data():
    data_f = open(DATA_PATH + 'data.pkl', 'rb')
    trainx, trainy = cPickle.load(data_f)
    validx, validy = cPickle.load(data_f)
    testx, testy = cPickle.load(data_f)
    return (trainx, trainy), (validx, validy), (testx, testy)


def load_word_id(max_word_count):
    d = {}
    for wn in open('word_count.txt'):
        if len(d) >= max_word_count:
            break
        d[wn.split()[0]] = len(d)
    return d


def load_tag_id():
    d = {}
    for tn in open(DATA_PATH + 'tag_count.txt'):
        d[tn.split()[0]] = len(d)
    return d
