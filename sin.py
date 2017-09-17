import argparse
from collections import OrderedDict
import cPickle
import sys
import time
from datetime import datetime

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from data.util import load_data, prepare_data
from util import *


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def init_params(options):
    """
    Global (not HLSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    Wemb = cPickle.load(open('./data/wv.pkl', 'rb'))
    Wemb.append(numpy.random.randn(options['dim_proj']))
    params['Wemb'] = numpy.array(Wemb, dtype='float32')

    params = param_init_hlstm(options, params)
    params = param_init_cnn(options, params)

    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'], options['ydim']).astype(config.floatX)
    params['U_cnn'] = 0.01 * numpy.random.randn(options['dim_proj'], options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def init_tparams(params):
    # from numpy objects to theano shared objects
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_hlstm(options, params):
    for h in ['h1', 'hatt', 'cnn_h1', 'cnn_hatt']:
        params['lstm_W_' + h] = numpy.concatenate([ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj'])], axis=1)
        params['lstm_U_' + h] = numpy.concatenate([ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj'])], axis=1)
        # peephole for input, forget, and output gate
        params['lstm_W_pi_' + h] = ortho_weight(options['dim_proj'])
        params['lstm_W_pf_' + h] = ortho_weight(options['dim_proj'])
        params['lstm_W_po_' + h] = ortho_weight(options['dim_proj'])

        b = numpy.zeros((4 * options['dim_proj'],), dtype=config.floatX)
        # increase bias for the forget gate
        b[options['dim_proj'] : 2 * options['dim_proj']] += 1.0
        params['lstm_b_' + h] = b

    for attl in ['a', 'cnn']:
        params[attl + 'att_W'] = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)

        params[attl + 'att_cand_ph'] = 0.01 * numpy.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
        params[attl + 'att_cand_ch'] = 0.01 * numpy.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
        params[attl + 'att_cand_b'] = 0.01 * numpy.random.randn(options['dim_proj'],).astype(config.floatX)

        params[attl + 'att_i_ph'] = 0.01 * numpy.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
        params[attl + 'att_i_ch'] = 0.01 * numpy.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
        params[attl + 'att_i_b'] = 0.01 * numpy.random.randn(options['dim_proj'],).astype(config.floatX)

    return params


def param_init_cnn(options, params):
    params['cnn_filter_20'] = 0.01 * numpy.random.randn(options['dim_proj'], options['dim_proj']).astype('float32')
    params['cnn_filter_21'] = 0.01 * numpy.random.randn(options['dim_proj'], options['dim_proj']).astype('float32')
    params['cnn_b_2'] = 0.01 * numpy.random.randn(options['dim_proj']).astype('float32')
    return params


def lstm_layer(tparams, dx, mask, options, hierarchy, attl):

    n_sentence = dx.shape[0]
    s_maxlen = dx.shape[1]

    def _slice(_x, n, dim):
        if _x.ndim == 1:
            return _x[n * dim : (n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        print 'hierarchy=', hierarchy

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

    
    def _h1_step(i, h_, c_, x, m):
        return _step(m[:,i], x[:,i,:], h_, c_)

    def _att_step(ch, cm, ph, pm):
        cand = tensor.tanh(tensor.dot(ph, tparams[attl + 'att_cand_ph'])[:,None,:] +\
                        tensor.dot(ch, tparams[attl + 'att_cand_ch'])[None,:,:] +\
                        tparams[attl + 'att_cand_b'])
        i = tensor.nnet.sigmoid(tensor.dot(ph, tparams[attl + 'att_i_ph'])[:,None,:] +\
                        tensor.dot(ch, tparams[attl + 'att_i_ch'])[None,:,:] +\
                        tparams[attl + 'att_i_b'])
        catts = (cand * i * pm[:,None,None]).sum(axis=0)
        return ch, cm, catts


    if hierarchy in ('h1', 'cnn_h1'):
        
        hbk = hierarchy

        i = tensor.arange(s_maxlen)
        hierarchy = 'hatt' if hbk == 'h1' else 'cnn_hatt'
        act = tensor.dot(dx, tparams['lstm_W_' + hierarchy]) + tparams['lstm_b_' + hierarchy]
        rv1, up1 = theano.scan(_h1_step,
                        sequences=[i],
                        outputs_info=[tensor.alloc(numpy_floatX(0.), n_sentence, options['dim_proj']),
                                      tensor.alloc(numpy_floatX(0.), n_sentence, options['dim_proj'])],
                        non_sequences=[act, mask],
                        name='lstm_' + hierarchy,
                        n_steps=s_maxlen)

        rv2, up2 = theano.scan(_att_step,
                        sequences=[rv1[0].dimshuffle((1,0,2)), mask],
                        outputs_info=[tensor.alloc(numpy_floatX(0.), s_maxlen, options['dim_proj']),
                                      tensor.alloc(numpy_floatX(0.), s_maxlen),
                                      None],
                        name='lstm_att',
                        n_steps=n_sentence)

        hierarchy = hbk
        act = tensor.dot(dx, tparams['lstm_W_' + hierarchy]) +\
                        tensor.dot(rv2[2], tparams[attl + 'att_W']) +\
                        tparams['lstm_b_' + hierarchy]

        rval, updates = theano.scan(_h1_step,
                        sequences=[i],
                        outputs_info=[tensor.alloc(numpy_floatX(0.), n_sentence, options['dim_proj']),
                                      tensor.alloc(numpy_floatX(0.), n_sentence, options['dim_proj'])],
                        non_sequences=[act, mask],
                        name='lstm_' + hierarchy,
                        n_steps=s_maxlen)
        return rval[0][-1]


def cnn_layer(tparams, dx, dm, width):

    dx = dx.dimshuffle((1,0,2))
    dm = dm.dimshuffle((1,0))

    def _step(i):
        x = tparams['cnn_b_%d' % width]
        m = 1
        for j in range(width):
            x += tensor.dot(dx[i + j], tparams['cnn_filter_%d%d' % (width, j)])
            m *= dm[i + j]
        x = tensor.tanh(x)
        return x, m

    maxlen = dx.shape[0]
    i = tensor.arange(maxlen - width + 1)
    rv, up = theano.scan(_step,
                        sequences=[i],
                        n_steps=maxlen - width + 1
                        )

    return rv[0].dimshuffle((1,0,2)), rv[1].dimshuffle((1,0))


def adadelta(tparams, grads, x, mask, y, cost, err):
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

    f_grad_shared = theano.function([x, mask, y], [cost, err], updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([], [], updates=ru2up + param_up, name='adadelta_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams()
    use_noise = theano.shared(numpy_floatX(1.))

    x = tensor.matrix('x', dtype='int32')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int32')

    n_sentence = x.shape[0]
    s_maxlen = x.shape[1]

    # hlstm_h1: calculate sentence vectors 
    emb = tparams['Wemb'][x.flatten()].reshape([n_sentence, s_maxlen, options['dim_proj']])
    emb_cnn, mask_cnn = cnn_layer(tparams, emb, mask, 2)

    sentence_vectors = lstm_layer(tparams, emb, mask, options, 'h1', 'a')
    sentence_vectors_cnn = lstm_layer(tparams, emb_cnn, mask_cnn, options, 'cnn_h1', 'cnn')
    print 'sentence_vectors.ndim = ', sentence_vectors.ndim

    proj = dropout_layer(sentence_vectors, use_noise, trng)
    proj_cnn = dropout_layer(sentence_vectors_cnn, use_noise, trng)

    pred_prob = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) +\
                                    tensor.dot(proj_cnn, tparams['U_cnn']) + tparams['b'])
    pred = pred_prob.argmax(axis=1)

    cost = (-tensor.log(pred_prob[tensor.arange(n_sentence), y.flatten()] + 1e-6) * mask[:,0].flatten()).sum() / mask[:,0].sum()
    err = 1.0 - (tensor.eq(pred, y) * mask[:,0]).sum() / mask[:,0].sum()

    f_pred = theano.function([x, mask], pred)

    return use_noise, x, mask, y, f_pred, cost, err

def pred_acc(f_pred, data, iterator):
    acc, N = 0.0, 0.0
    for _, index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in index], numpy.array(data[1])[index])
        preds = f_pred(x, mask)
        acc += ((preds == y) * mask[:,0]).sum()
        N += mask[:,0].sum()
    return acc / N


def train_lstm(
    dim_proj,  # word embeding dimension and LSTM number of hidden units.
    max_epochs,  # The maximum number of epoch to run
    n_words,  # Vocabulary size
    validFreq,  # Compute the validation error after this number of update.
    batch_size,  # The batch size during training.
    valid_batch_size,  # The batch size used for validation/test set.
):

    # Model options
    model_options = locals()
    print "model options", model_options

    print 'Loading data'
    train, valid, test = load_data()

    ydim = max([max(y) for y in train[1]]) + 1
    model_options['ydim'] = ydim

    print 'Building model'
    params = init_params(model_options)
    tparams = init_tparams(params)

    (use_noise, x, mask, y, f_pred, cost, err) = build_model(tparams, model_options)

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size, shuffle=False)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size, shuffle=False)

    print 'grads'
    grads = tensor.grad(cost, wrt=tparams.values())

    print 'adadelta'
    f_grad_shared, f_update = adadelta(tparams, grads, x, mask, y, cost, err)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    # best valid error and corresponding test error
    best_valid_acc = [0, 0, 0, 0]

    uidx = 0  # the number of update done
    start_time = time.time()

    frecord = open('record/da-sin-conv.csv', 'w')
    for eidx in xrange(max_epochs):
        kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

        for _, train_index in kf:
            uidx += 1
            use_noise.set_value(1.)

            x = [train[0][t] for t in train_index]
            y = [train[1][t] for t in train_index]
            x, mask, y = prepare_data(x, y)

            cost, err = f_grad_shared(x, mask, y)
            print 'epoch=%d, batch_id=%d, cost=%.4f, err=%.4f, time=%.1f' % (eidx, _, cost, err, time.time() - start_time)

            f_update()

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'bad cost detected: ', cost
                return 1., 1., 1.

            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                kf_train_sample = get_minibatches_idx(len(train[0]), valid_batch_size, shuffle=True, sample_ratio=0.02)
                train_acc = pred_acc(f_pred, train, kf_train_sample)
                valid_acc = pred_acc(f_pred, valid, kf_valid)
                test_acc = pred_acc(f_pred, test, kf_test)

                curr = (train_acc, valid_acc, test_acc, eidx)
                frecord.write('%.3f,%.3f,%.3f,%d\n' % curr)
                frecord.flush()
                if valid_acc > best_valid_acc[1]:
                    best_valid_acc = curr
                print 'curr:\ttrain:%.3f\tvalid:%.3f\ttest:%.3f\teidx:%d' % curr
                print 'best_valid_acc:\ttrain:%.3f\tvalid:%.3f\ttest:%.3f\teidx:%d' % best_valid_acc

    frecord.close()
    return best_valid_acc[:3]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-dim_proj', type=int, default=128, help='word embeding dimension and HLSTM number of hidden units.')
    ap.add_argument('-max_epochs', type=int, default=5000, help='The maximum number of epoch to run')
    ap.add_argument('-n_words', type=int, default=10000, help='Vocabulary Size')
    ap.add_argument('-validFreq', type=int, default=10, help='Compute the validation error after this number of update.')
    ap.add_argument('-batch_size', type=int, default=10, help='The batch size during training.')
    ap.add_argument('-valid_batch_size', type=int, default=30, help='The batch size used for validation/test set.')
    if theano.config.floatX == 'float32':
        args = vars(ap.parse_args())
        train_lstm(**args)
    else:
        print 'error, use the following command'
        print 'THEANO_FLAGS="floatX=float32" python hlstm.py'
