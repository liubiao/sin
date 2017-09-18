import argparse
from collections import OrderedDict
import cPickle
import sys
import os
import time
from datetime import datetime

import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from data.util import load_data, prepare_data
from util import *
import numpy as np

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def init_tparams(params):
    # from np objects to theano shared objects
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def init_params(options):
    """
    Global (not HLSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()

    Wemb = cPickle.load(open('./data/wv.pkl', 'rb'))
    Wemb.append(np.random.randn(options['dim_proj']))

    params['Wemb'] = np.array(Wemb, dtype='float32')
    params = param_init_hlstm(options, params)

    # classifier
    params['Uq'] = 0.01 * np.random.randn(options['dim_proj']).astype(config.floatX)
    params['Ua'] = 0.01 * np.random.randn(options['dim_proj']).astype(config.floatX)
    params['b'] = numpy_floatX(0)

    return params


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_hlstm(options, params):
    for h in ['h1', 'h2']:
        params['lstm_W_' + h] = np.concatenate([ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj'])], axis=1)
        params['lstm_U_' + h] = np.concatenate([ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj'])], axis=1)
        # peephole for input, forget, and output gate
        params['lstm_W_pi_' + h] = ortho_weight(options['dim_proj'])
        params['lstm_W_pf_' + h] = ortho_weight(options['dim_proj'])
        params['lstm_W_po_' + h] = ortho_weight(options['dim_proj'])

        b = np.zeros((4 * options['dim_proj'],), dtype=config.floatX)
        # increase bias for the forget gate
        b[options['dim_proj'] : 2 * options['dim_proj']] += 1.0
        params['lstm_b_' + h] = b

    params['att_W'] = np.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)

    params['att_cand_ph'] = 0.01 * np.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
    params['att_cand_ch'] = 0.01 * np.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
    params['att_cand_b'] = 0.01 * np.random.randn(options['dim_proj'],).astype(config.floatX)

    params['att_i_ph'] = 0.01 * np.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
    params['att_i_ch'] = 0.01 * np.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
    params['att_i_b'] = 0.01 * np.random.randn(options['dim_proj'],).astype(config.floatX)
    
    return params



def lstm_layer(tparams, dx, dm, options, hierarchy, att=None):

    def _slice(_x, n, dim):
        return _x[:, n * dim:(n + 1) * dim]

    def _step(x_, m_, h_, c_):

        preact = tensor.dot(h_, tparams['lstm_U_' + hierarchy]) + x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']) + tensor.dot(c_, tparams['lstm_W_pi_' + hierarchy]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']) + tensor.dot(c_, tparams['lstm_W_pf_' + hierarchy]))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']) + tensor.dot(c, tparams['lstm_W_po_' + hierarchy]))
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    act = tensor.dot(dx, tparams['lstm_W_' + hierarchy]) + tparams['lstm_b_' + hierarchy]
    if att:
        act += att

    maxlen, sn = dx.shape[0], dx.shape[1]
    rv, up = theano.scan(_step,
                    sequences=[act, dm],
                    outputs_info=[tensor.alloc(numpy_floatX(0.), sn, options['dim_proj']),
                                  tensor.alloc(numpy_floatX(0.), sn, options['dim_proj'])],
                    name='lstm_' + hierarchy,
                    n_steps=maxlen)
    return rv[0]


def att_layer(tparams, ph, pm, ch):
    cand = tensor.tanh(tensor.dot(ph, tparams['att_cand_ph'])[:,None,:,:] +\
                    tensor.dot(ch, tparams['att_cand_ch'])[None,:,:,:] +\
                    tparams['att_cand_b'])
    i = tensor.nnet.sigmoid(tensor.dot(ph, tparams['att_i_ph'])[:,None,:,:] +\
                    tensor.dot(ch, tparams['att_i_ch'])[None,:,:,:] +\
                    tparams['att_i_b'])
    catts = (cand * i * pm[:,None,:,None]).sum(axis=0) 
    catts = tensor.dot(catts, tparams['att_W'])
    return catts



def build_model(tparams, options):
    trng = RandomStreams()

    use_noise = theano.shared(numpy_floatX(1.))

    q = tensor.matrix('q', dtype='int32')
    qm = tensor.matrix('qm', dtype='float32')
    qmaxlen, qn = q.shape[0], q.shape[1]

    a = tensor.matrix('a', dtype='int32')
    am = tensor.matrix('am', dtype='float32')
    amaxlen, an = a.shape[0], a.shape[1]

    y = tensor.vector('y', dtype='int32')

    # lstm1: calculate sentence vectors 
    qemb = tparams['Wemb'][q.flatten()].reshape([qmaxlen, qn, options['dim_proj']])
    qx = lstm_layer(tparams, qemb, qm, options, 'h1')

    aemb = tparams['Wemb'][a.flatten()].reshape([amaxlen, an, options['dim_proj']])
    ax = lstm_layer(tparams, aemb, am, options, 'h1')

    # att layer
    atta = att_layer(tparams, qx, qm, ax)
    attq = att_layer(tparams, ax, am, qx)

    # lstm2
    projq = lstm_layer(tparams, qemb, qm, options, 'h2', attq)
    proja = lstm_layer(tparams, aemb, am, options, 'h2', atta)

    projq = (projq * qm[:,:,None]).sum(axis=0) / (qm.sum(axis=0)[:,None] + 1e-6)
    proja = (proja * am[:,:,None]).sum(axis=0) / (am.sum(axis=0)[:,None] + 1e-6)

    projq = dropout_layer(projq, use_noise, trng)
    proja = dropout_layer(proja, use_noise, trng)

    pred_prob = tensor.nnet.sigmoid(tensor.dot(projq, tparams['Uq']) +\
                                    tensor.dot(proja, tparams['Ua']) +\
                                    tparams['b'])


    cost = -(y * tensor.log(pred_prob) + (1 - y) * tensor.log(1 - pred_prob)).sum() / an

    f_pred_prob = theano.function([q, qm, a, am], pred_prob)

    return use_noise, q, qm, a, am, y, f_pred_prob, cost


def adadelta(tparams, grads, q, qm, a, am, y, cost):
    # [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning Rate Method*, arXiv:1212.5701.

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inputs=[q, qm, a, am, y], outputs=cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([], [], updates=ru2up + param_up, name='adadelta_f_update')

    return f_grad_shared, f_update

def evaluate(f_pred_prob, data, iterator):
    ys, ps, qi = [], [], []
    for index in iterator:
        q = [data[0][t] for t in index]
        a = [data[1][t] for t in index]
        y = [data[2][t] for t in index]
        i = [data[3][t] for t in index]
        qx, qm = prepare_data(q)
        ax, am = prepare_data(a)
        p = f_pred_prob(qx, qm, ax, am)
        ys = ys + y
        qi = qi + i
        ps = ps + p.tolist()
    ix = range(len(ys))
    ix.sort(key=lambda i: qi[i] + ps[i], reverse=True)
    mapx = mrr = nx = ap = n = tn = 0.0
    oldq = -1
    for i in ix:
        if qi[i] != oldq:
            if tn > 0:
                mapx += ap / tn
                nx += 1.0
            oldq = qi[i]
            ap = n = tn = 0.0
        n += 1.0
        if ys[i]:
            tn += 1.0
            ap += tn / n
            if tn == 1.0:
                mrr += 1.0 / n
    if tn > 0:
        mapx += ap / tn
        nx += 1.0
    return mapx / nx, mrr / nx


def train_lstm(
    dim_proj,  # word embeding dimension and LSTM number of hidden units.
    max_epochs,  # The maximum number of epoch to run
    validFreq,  # Compute the validation error after this number of update.
    batch_size,  # The batch size during training.
    valid_batch_size,  # The batch size used for validation/test set.
):

    # Model options
    options = locals()
    print "model options", options

    print 'Loading data'
    train, valid, test = load_data()

    print 'Building model'
    params = init_params(options)
    tparams = init_tparams(params)

    use_noise, q, qm, a, am, y, f_pred_prob, cost = build_model(tparams, options)

    print 'model done'

    print 'grads'
    grads = tensor.grad(cost, wrt=tparams.values())

    lr = tensor.scalar(name='lr')
    print 'adadelta'
    f_grad_shared, f_update = adadelta(tparams, grads, q, qm, a, am, y, cost)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    best_valid_map = (0,0,0,0,0,0,0)
    kf_valid = get_minibatches_idx(valid, valid_batch_size, shuffle=False)
    kf_test = get_minibatches_idx(test, valid_batch_size, shuffle=False)

    uidx = 0  # the number of update done
    start_time = time.time()
    if not os.path.exists('record'):
        os.mkdir('record')
    frecord = open('record/as-sin.csv', 'w')
    for eidx in range(max_epochs):
        # Get new shuffled index for the training set.
        kf = get_minibatches_idx(train, batch_size)
        
        for bid, train_index in enumerate(kf):

            uidx += 1
            use_noise.set_value(1.)

            # Select the random examples for this minibatch
            q = [train[0][t] for t in train_index]
            a = [train[1][t] for t in train_index]
            y = [train[2][t] for t in train_index]

            # Get the data in np.ndarray format
            # This swap the axis!
            # Return something of shape (minibatch maxlen, n samples)
            q, qm = prepare_data(q)
            a, am = prepare_data(a)

            cost = f_grad_shared(q, qm, a, am, y)
            print 'epoch=%d, batch_id=%d, cost=%.4f, time=%.1f' % (eidx, bid, cost, time.time() - start_time)

            f_update()

            if np.isnan(cost) or np.isinf(cost):
                print 'bad cost detected: ', cost
                return 1., 1., 1.

            if np.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                kf_train_sample = get_minibatches_idx(train, valid_batch_size, shuffle=True, sample_ratio=0.1)
                train_map, train_mrr = evaluate(f_pred_prob, train, kf_train_sample)
                valid_map, valid_mrr = evaluate(f_pred_prob, valid, kf_valid)
                test_map, test_mrr = evaluate(f_pred_prob, test, kf_test)

                frecord.write('%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (train_map, valid_map, test_map, train_mrr, valid_mrr, test_mrr))
                frecord.flush()

                if valid_map >= best_valid_map[2]:
                    best_valid_map = (train_map, train_mrr, valid_map, valid_mrr, test_map, test_mrr, eidx)

                print 'curr\ttrain:%.3f,%.3f\tvalid:%.3f,%.3f\ttest:%.3f,%.3f' % \
                        (train_map, train_mrr, valid_map, valid_mrr, test_map, test_mrr)
                print 'best_valid_map\ttrain:%.3f,%.3f\tvalid:%.3f,%.3f\ttest:%.3f,%.3f\tepoch:%d' % best_valid_map


    end_time = time.time()
    frecord.close()

    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return best_valid_map


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-dim_proj', type=int, default=128, help='word embeding dimension and HLSTM number of hidden units.')
    ap.add_argument('-max_epochs', type=int, default=5000, help='The maximum number of epoch to run')
    ap.add_argument('-validFreq', type=int, default=10, help='Compute the validation error after this number of update.')
    ap.add_argument('-batch_size', type=int, default=10, help='The batch size during training.')
    ap.add_argument('-valid_batch_size', type=int, default=30, help='The batch size used for validation/test set.')
    if theano.config.floatX == 'float32':
        args = vars(ap.parse_args())
        train_lstm(**args)
    else:
        print 'error, use the following command'
        print 'THEANO_FLAGS="floatX=float32" python hlstm.py'
