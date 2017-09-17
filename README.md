# How to run

## Answer Selection
1. Clone this repository, checkout to branch as-sin (SIN model) or as-sin-conv (SIN-CONV model).
2. Download WikiQA from [http://aka.ms/WikiQA](http://aka.ms/WikiQA). Decompress the file, and put WikiQA-train.txt, WikiQA-test.txt, WikiQA-dev.txt into the directory <code>sin/data/</code>
3. Download glove.6B.zip from [http://nlp.stanford.edu/projects/glove/](http://nlp.stanford.edu/projects/glove/). Decompress the file, and put glove.6B.100d.txt into the directory <code>sin/data/glove/</code>
4. Go to <code>sin/data</code>, run <code>python process.py</code> for data preprocessing.
5. Go to <code>sin/</code>, run <code>./run.sh</code>


## Dialogue Act Analysis
1. Clone this repository, checkout to branch da-sin (SIN) or da-sin-conv (SIN-CONV) or da-sin-ld (SIN-LD) or da-sin-conv-ld (SIN-CONV-LD)
2. Download swda.zip from [http://compprag.christopherpotts.net/swda.html](http://compprag.christopherpotts.net/swda.html). Decompress the file, and put <code>swda/</code> into <code>sin/data/</code>
3. Download glove.6B.zip from [http://nlp.stanford.edu/projects/glove/](http://nlp.stanford.edu/projects/glove/). Decompress the file, and put glove.6B.100d.txt into the directory <code>sin/data/glove/</code>
4. Go to <code>sin/data</code>, run <code>python preprocess.py</code> for data preprocessing.
5. Go to <code>sin/</code>, run <code>./run.sh</code>
