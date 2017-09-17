'''

1. Code from "http://compprag.christopherpotts.net/swda.html" are used in this script.
2. Note that there are 42 tags in total. The tag "+" is interpreted as continuation of the previous utterance, see http://web.stanford.edu/~jurafsky/ws97/manual.august1.html, and should not be treated as a valid tag. 
3. The train, test split can be found at: http://web.stanford.edu/~jurafsky/ws97/

'''

import os
import sys
import re
from collections import Counter, OrderedDict
import numpy as np
import cPickle

import nltk
from swda import CorpusReader

from util import load_word_id, load_tag_id

def tackle_utterance(u):
    u.text = u.text.lower()
    u.text = re.sub(r'\*', ' * ', u.text)
    u.text = re.sub(r'\(\(.*?\)\)', ' __BRACKET__ ', u.text)
    u.text = re.sub(r'<<.*?>>', ' __ANGLE_BRACKET__ ', u.text)
    u.text = re.sub(r'<(?P<sound>.*?)>', ' __\g<sound>__ __SOUND__ ', u.text)
    words = []
    raw_w  = ' '.join(u.text_words(filter_disfluency=True))
    for w in nltk.word_tokenize(raw_w):
        words.append(w)
        if len(w) > 1 and w[-1] == '-' and w[-2].isalpha():
            words.append('__INCOMPLETE__')
    return words


def word_tag_count():
    splits = load_data_split()
    print 'Counting words and tags...'
    wc, tc = Counter(), Counter()
    cr = CorpusReader('swda')
    for i, t in enumerate(cr.iter_transcripts()):
        sid = split_id(t.swda_filename, splits)
        # only the train(sid=0) and valid(sid=1) set are counted
        # test(sid=2) set and others(sid=3) are not counted
        if sid < 2:
            for u in t.utterances:
                tag = u.damsl_act_tag()
                # if tag is '+', it is regarded as continuation of the previous utterance
                # see: http://web.stanford.edu/~jurafsky/ws97/manual.august1.html
                if tag != '+':
                    tc[tag] += 1
                for w in tackle_utterance(u):
                    wc[w] += 1
    tf = open('tag_count.txt', 'w')
    for t, n in tc.most_common():
        tf.write('%s %d\n' % (t, n))
    tf.close()
    wf = open('word_count.txt', 'w')
    for w, n in wc.most_common():
        wf.write('%s %d\n' % (w, n))
    wf.close()
    print 'done'
    

def load_data_split():
    train_valid = np.array([line.strip() for line in open('train.list')])
    test = [line.strip() for line in open('test.list')]
    tid = range(len(train_valid))
    np.random.seed(131)
    np.random.shuffle(tid)
    train = sorted(train_valid[tid[:1085]].tolist())
    valid = sorted(train_valid[tid[1085:]].tolist())
    return train, valid, test


def split_id(name, splits):
    nid = re.split(r'_|\.', name)[-3]
    for i, s in enumerate(splits):
        if nid in s:
            return i
    return 3


def gen_data(max_word_count=10000):
    print 'generating train/valid/test/ dataset...'

    wid_map = load_word_id(max_word_count)
    tagid_map = load_tag_id()

    # preparing data
    all_x, all_y = [], []
    # train/valid/test split
    ids = [[], [], [], []]
    splits = load_data_split()

    cr = CorpusReader('swda')
    for i, t in enumerate(cr.iter_transcripts()):
        sid = split_id(t.swda_filename, splits)
        ids[sid].append(i)
        tx, ty = [], []
        for u in t.utterances:
            x = []
            # the tag_id_map do not contains tag "+", and thus "+" is mapped to -1
            y = tagid_map.get(u.damsl_act_tag(), -1)
            for w in tackle_utterance(u):
                wid = wid_map.get(w)
                wid = max_word_count if wid is None else wid
                x.append(wid)
            if y != -1:
                tx.append(x)
                ty.append(y)
        all_x.append(tx)
        all_y.append(ty)


    def _fetch(i):
        return [all_x[t] for t in i], [all_y[t] for t in i]

    train_x, train_y = _fetch(ids[0])
    valid_x, valid_y = _fetch(ids[1])
    test_x, test_y = _fetch(ids[2])

    data_f = open('data.pkl', 'wb')
    cPickle.dump((train_x, train_y), data_f, 2)
    cPickle.dump((valid_x, valid_y), data_f, 2)
    cPickle.dump((test_x, test_y), data_f, 2)
    data_f.close()


def get_word_vectors(max_word_count, dimension):

    print 'getting word vectors...'
    wv = OrderedDict()
    for line in open('word_count.txt'):
        w, n = line.split()
        v = np.random.randn(dimension)
        wv[w] = v / np.sqrt((v * v).sum())
        if len(wv) >= max_word_count:
            break

    fwemb = open('./glove/glove.6B.%dd.txt' % dimension)
    for line in fwemb:
        v = line.split()
        if v[0] in wv:
            wv[v[0]] = [float(t) for t in v[1:]]
    cPickle.dump(wv.values(), open('wv.pkl', 'wb'))


def preprocess():
    word_tag_count()
    gen_data(max_word_count=10000)
    get_word_vectors(10000, 100)

if __name__ == '__main__':
    preprocess()
