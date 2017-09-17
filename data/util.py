import numpy
import cPickle


def prepare_data(ss):
    maxlen = max([len(s) for s in ss])

    # truncate to 40 words
    maxlen = 40 if maxlen > 40 else maxlen

    x = numpy.zeros(shape=(maxlen, len(ss)), dtype='int32')
    m = numpy.zeros(shape=(maxlen, len(ss)), dtype='float32')
    for i, s in enumerate(ss):
        if len(s) > 40:
            s = s[:40]
        x[:len(s), i] = s
        m[:len(s), i] = 1
    return x, m


def load_data():
    train = cPickle.load(open('data/train.pkl', 'rb'))
    dev = cPickle.load(open('data/dev.pkl', 'rb'))
    test = cPickle.load(open('data/test.pkl', 'rb'))
    return train, dev, test
