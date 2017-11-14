# EmbeddingVectorizer

This repository contains an implementation of an sklearn Vectorizer that
 produces document embeddings via the SIF algorithm described in
 [Arora, Sanjeev, Yingyu Liang, and Tengyu Ma. "A simple but tough-to-beat baseline for sentence embeddings." (2016)](https://openreview.net/pdf?id=SyK00v5xx)


## Install
``$ pip install -r requirements.txt``

## Get started

sts_benchmark.ipynb reproduces the [results](http://www.aclweb.org/anthology/S/S17/S17-2001.pdf) on the [STSbenchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)
dataset. The benchmark uses [spacy](spacy.io) to obtain word vectors for the words
in the corpus, but the vectorizer accepts word vectors from any source (e.g. [GloVe](https://nlp.stanford.edu/projects/glove/)).
Unigram probabilities p(w) are obtained from [enwiki_vocab_min200.txt](https://github.com/PrincetonML/SIF/tree/master/auxiliary_data)
used in the original paper or by the use of the [wordfreq](https://pypi.python.org/pypi/wordfreq) package.

## Example
```
vectorizer = EmbeddingVectorizer(
    tokenizer=lambda doc: doc.split(),
    word_vectorizer=lambda word: word_vectors[word],
    word_freq=lambda word: word_frequencies[word],
    weighted=True,
    remove_components=1,
    lowercase=True)

vectorizer.fit(docs_train)

vectorizer.transform(docs_test)
```

## Sources

### STSbenchmark
- **Description:** http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark
- **Dataset:** http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
- **Paper:** http://www.aclweb.org/anthology/S/S17/S17-2001.pdf

### SIF
- **Paper:** https://openreview.net/pdf?id=SyK00v5xx
- **Github:** https://github.com/PrincetonML/SIF
- **enwiki_vocab_min200.txt:** https://github.com/PrincetonML/SIF/tree/master/auxiliary_data
