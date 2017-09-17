#!/bin/bash

validFreq=10
batch_size=3
valid_batch_size=3
dim_proj=100
n_words=10000
max_epochs=30

cmd="python sin.py -validFreq $validFreq -valid_batch_size $valid_batch_size -batch_size $batch_size -dim_proj $dim_proj -n_words $n_words -max_epochs $max_epochs"

if [ $# -eq 0 ]; then
	#THEANO_FLAGS="mode=FAST_RUN,floatX=float32,exception_verbosity=high" $cmd
	THEANO_FLAGS="mode=FAST_RUN,floatX=float32" $cmd

elif [ "$1" = "gpu" ]; then
	export PATH="/usr/local/cuda/bin:${PATH}"
	export LD_LIBRARY_PATH="/usr/local/cuda/lib:${LD_LIBRARY_PATH}"
	export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
	THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32" $cmd
else
	echo "error"
fi
