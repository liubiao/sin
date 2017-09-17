#!/bin/bash

validFreq=10
batch_size=30
valid_batch_size=200
dim_proj=100
max_epochs=10

cmd="python sin.py -validFreq $validFreq -valid_batch_size $valid_batch_size -batch_size $batch_size -dim_proj $dim_proj -max_epochs $max_epochs"

THEANO_FLAGS="mode=FAST_RUN,floatX=float32" $cmd
