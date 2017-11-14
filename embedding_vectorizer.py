import six
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


class EmbeddingVectorizer(CountVectorizer):
    """Convert a collection of text documents to a matrix of document
    embeddings.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing steps.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token". The default
        regexp select tokens of 2 or more alphanumeric characters
        (punctuation is completely ignored and always treated as a token
        separator).

    word_vectorizer: callable or None (default)
        Override the word vector computation step during document embedding
        computation. The callable receives a word and should return a vector
        of size n_features.

    word_freq: callable or None (default)
        Override the word frequency computation step during document embedding
        computation. Only used if weighted is True. The callable receives a
        word and should return a frequency between 0 and 1.
        If None, all word receive a frequency of 0.

    weighted: boolean, default True
        If True, each word vector is weighted according to a / (a + freq)
        before averaging.

    remove_components: int, default 1
        If > 0, the final document embeddings are computed by removing the
        common component(s) from the document embeddings by subtracting the
        projections to their principal components.

    a: float, default 1e-3
        The weighting parameter used during word vector averaging. The best
        performance is usually achieved at a=1e-3 to a=1e-4.

    Attributes
    ----------
    common_components_ : array, shape = [n_features] or None
        The common components

    See also
    --------
    HashingVectorizer, TfidfVectorizer

    Notes
    -----
    Based on Arora, Sanjeev, Yingyu Liang, and Tengyu Ma.
    "A simple but tough-to-beat baseline for sentence embeddings." (2016).
    """
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, preprocessor=None,
                 tokenizer=None, stop_words=None, lowercase=True,
                 token_pattern=r"(?u)\b\w\w+\b", word_vectorizer=None,
                 word_freq=None, weighted=True, remove_components=1,
                 a=1e-3):

        super(EmbeddingVectorizer, self).__init__(
            input, encoding, decode_error, strip_accents, lowercase,
            preprocessor, tokenizer, stop_words, token_pattern)

        self.word_vectorizer = word_vectorizer
        self.word_freq = word_freq or (lambda w: 0.0)
        self.weighted = weighted
        self.remove_components = remove_components
        self.a = a

    def _average_sentence_vec(self, raw_documents):
        """Calculate the sentence vector by computing the (weighted) mean of
        the word vectors in the sentence
        """
        analyze = self.build_analyzer()

        sentence_vecs = []
        for doc in raw_documents:
            word_vecs = []
            for token in analyze(doc):
                token = token.lower() if self.lowercase else token
                wv = np.array(self.word_vectorizer(token))
                if self.weighted:
                    freq = self.word_freq(token)
                    wv *= self.a / (self.a + freq)
                word_vecs.append(wv)
            vs = np.array(word_vecs).mean(axis=0)
            sentence_vecs.append(vs)

        return np.array(sentence_vecs)

    def _compute_singular_vectors(self, X):
        svd = TruncatedSVD(n_components=self.remove_components, n_iter=7,
                           random_state=0).fit(X)
        self.common_components_ = svd.components_

    def _remove_common_components(self, X):
        u = self.common_components_
        return X - X.dot(u.transpose()).dot(u)

    def fit(self, raw_documents, y=None):
        """Learns the common component(s) from the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the common component(s) and return a document embedding
        matrix. This is equivalent to fit followed by transform, but more
        efficiently implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document embedding matrix.
        """
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        X = self._average_sentence_vec(raw_documents)

        if self.remove_components > 0:
            self._compute_singular_vectors(X)
            X = self._remove_common_components(X)

        return X

    def transform(self, raw_documents):
        """Transform documents to document embedding matrix removing the
        common components computed during fit (if specified).

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document embedding matrix.
        """
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        X = self._average_sentence_vec(raw_documents)

        if self.remove_components > 0:
            X = self._remove_common_components(X)

        return X
