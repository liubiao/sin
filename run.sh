#!/bin/bash

validFreq=10
batch_size=3
valid_batch_size=3
dim_proj=100
n_words=10000
max_epochs=30

cmd="python sin.py -validFreq $validFreq -valid_batch_size $valid_batch_size -batch_size $batch_size -dim_proj $dim_proj -n_words $n_words -max_epochs $max_epochs"

THEANO_FLAGS="mode=FAST_RUN,floatX=float32" $cmd
