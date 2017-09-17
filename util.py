import numpy
import theano
from theano import config
import theano.tensor as tensor
from collections import OrderedDict, defaultdict
import random

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(data, minibatch_size, shuffle=True, sample_ratio=1.0):
    if sample_ratio != 1.0:
        d = defaultdict(list)
        dy = defaultdict(int)
        for i, q in enumerate(data[3]):
            dy[q] += data[2][i]
            d[q].append(i)

        qid = range(data[3][-1] + 1)
        if shuffle:
            numpy.random.shuffle(qid)
        selected_id = []
        for q in qid:
            if dy[q] > 0:
                selected_id.append(q)
            if len(selected_id) >= len(qid) * sample_ratio:
                break
        res, tmp = [], []
        for q in selected_id:
            if len(tmp) >= minibatch_size:
                res.append(tmp)
                tmp = []
            tmp += d[q]
        if tmp:
            res.append(tmp)
    else:
        ids = range(len(data[0]))
        if shuffle:
            numpy.random.shuffle(ids)
        ids = ids[:int(sample_ratio * len(ids))]

        res, tmp = [], []
        for i in ids:
            if len(tmp) >= minibatch_size:
                res.append(tmp)
                tmp = []
            tmp.append(i)
        if tmp:
            res.append(tmp)
    return res


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params
