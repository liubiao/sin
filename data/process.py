#coding:utf8
from collections import Counter, OrderedDict
import cPickle
import numpy
from math import log
import re
import nltk
import math
import shutil

numberp = re.compile(r'^\d(\d|,)*(\.\d*)?$')

def cut(s):
    s = s.lower()
    ws = []
    try:
        tokens = nltk.tokenize.word_tokenize(s)
    except:
        print 'tokenize error:', s
        tokens = s.split()
    for w in tokens:
        if re.match(numberp, w):
            ws.append('__NUMBER__')
        ws.append(w)
    return ws


def filter(fin, fout):
    print 'filtering...'
    fin, fout = open(fin), open(fout, 'w')
    oldq, ta, tl = '', [], []
    for line in fin:
        q, a, l = line.split('\t')
        if q != oldq:
            if sum(tl) > 0:
                for ax, lx in zip(ta, tl):
                    fout.write('%s\t%s\t%d\n' % (oldq, ax, lx))
            oldq, ta, tl = q, [], []
        ta.append(a)
        tl.append(int(l))
    if sum(tl) > 0:
        for ax, lx in zip(ta, tl):
            fout.write('%s\t%s\t%d\n' % (oldq, ax, lx))
    fin.close()
    fout.close()


def collect_all_sentences():
    print 'collecting sentences...'
    s = []
    for name in ['train.txt', 'dev.txt', 'test.txt']:
        for line in open(name):
            q, a, l = line.split('\t')
            if not q in s:
                s.append(q)
            if not a in s:
                s.append(a)
    open('all.txt', 'w').write('\n'.join(s))


def tfdf(dimension):
    words = set([line.split()[0] for line in open('./glove/glove.6B.%dd.txt' % dimension)])
    words.add('__NUMBER__')
    print 'tfdf...'
    tf, df = Counter(), Counter()
    for line in open('all.txt'):
        ws = cut(line)
        tf.update(ws)
        df.update(set(ws))

    with open('tf.txt', 'w') as ftf:
        for w, n in tf.most_common():
            # if n == 1:
                # break
            if w in words:
                ftf.write('%s\t%d\n' % (w, n))

    with open('df.txt', 'w') as fdf:
        for w, n in df.most_common():
            fdf.write('%s\t%d\n' % (w, n))


def get_word_vectors(dimension):
    print 'getting word vectors...'
    wv = OrderedDict()
    for line in open('tf.txt'):
        w, n = line.split()
        v = numpy.random.randn(dimension)
        wv[w] = v / math.sqrt((v * v).sum())
    fwemb = open('./glove/glove.6B.%dd.txt' % dimension)
    for line in fwemb:
        v = line.split()
        if v[0] in wv:
            wv[v[0]] = [float(t) for t in v[1:]]
    cPickle.dump(wv.values(), open('wv.pkl', 'wb'))


def load_w2id():
    w2id = {}
    for line in open('tf.txt'):
        w2id[line.split()[0]] = len(w2id)
    return w2id


def load_idf():
    df = {}; N = 0
    for line in open('df.txt'):
        w, n = line.split()
        df[w] = int(n)
        N += int(n)
    idf = {}
    for w, n in df.items():
        idf[w] = log(1.0 * N / n)
    return idf


def features(qline, aline, idf, stopwords):
    q, a = qline.split(), aline.split()
    wn, wwn = 0, 0
    aset = set(a)
    for w in q:
        if (not w in stopwords) and w in aset:
            wn += 1
            wwn += idf[w]
    return numpy.array([len(q), len(a), wn, wwn], dtype='float32')


def get_data():
    print 'getting data...'
    w2id = load_w2id()
    idf = load_idf()
    stopwords = set([line.strip() for line in open('stopwords.txt')])
    for name in ['train', 'dev', 'test']:
        q, a, l, qi, f = [], [], [], [], []
        i = -1 
        oldq = ''
        for line in open(name + '.txt'):
            tq, ta, tl = line.lower().strip().split('\t')
            if tq != oldq:
                i += 1
                oldq = tq
                vq = [w2id.get(w, len(w2id)) for w in cut(tq)]
            va = [w2id.get(w, len(w2id)) for w in cut(ta)]
            q.append(vq)
            a.append(va)
            l.append(int(tl))
            qi.append(i)
            f.append(features(tq, ta, idf, stopwords))

        cPickle.dump((q, a, l, qi, f), open(name + '.pkl', 'wb'))

def main():
    shutil.copy('WikiQA-train.txt', 'train.txt')
    filter('WikiQA-dev.txt', 'dev.txt')
    filter('WikiQA-test.txt', 'test.txt')
    collect_all_sentences()
    dimension = 100
    tfdf(dimension)
    get_word_vectors(dimension)
    get_data()

if __name__ == '__main__':
    main()
