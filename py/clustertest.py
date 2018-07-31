"""
Cluster test cases.
@version 2018.07.30 Yun Hua
"""

import sys
import os
import unittest
import scipy as sp
import nltk.stem
import sklearn.datasets  # for 20Newsgroups, also gets http://mlcomp.org/datasets/379
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def dist_raw(v1, v2):
    """
    A primary distance function using Euclidean norm.
    """
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())


def dist_norm(v1, v2):
    """
    Uses norm function to calculate the vector distance instead.
    """
    v1_norm = v1 / sp.linalg.norm(v1.toarray())
    v2_norm = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_norm - v2_norm
    return sp.linalg.norm(delta.toarray())


class StemmedTfidfVectorizer(TfidfVectorizer):
    """
    A reasonable way to extract a compact vector from a noisy
    textual post: TF-IDF
    """
    english_stemmer = nltk.stem.SnowballStemmer('english')

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (StemmedTfidfVectorizer.english_stemmer.stem(w) for w in analyzer(doc))


class ClusterTest(unittest.TestCase):
    """
    Test clustering test cases.
    """

    def test_vectorizer(self):
        vectorizer = CountVectorizer(min_df=1, stop_words='english')
        print('sklearn CountVectorizer structure: \n {0}'.format(vectorizer))
        content = ['How to develope a handful server',
                   'Good server is under develope']

        # Learn the vocabulary dictionary and return term-document matrix.
        X_train = vectorizer.fit_transform(content)
        print(vectorizer.get_feature_names())

        new_txt = 'Develope good server'
        new_txt_vec = vectorizer.transform([new_txt])
        for i, txt in enumerate(content):
            txt_vec = X_train.getrow(i)
            d = dist_raw(txt_vec, new_txt_vec)
            print('== Txt {0:d} with dist={1:.2f}: {2:s}'.format(i, d, txt))

    def test_enhance_vectorizer(self):
        """
        Vectorize using NLTK's stemmer and stop words:
        term frequency - inverse document frequency (TF-IDF)
        """
        data_path = os.path.join(sys.path[0], 'data/txt')
        posts = []
        for f in os.listdir(data_path):
            with open(os.path.join(data_path, f)) as fin:
                posts.append(fin.read())

        # Stop words: removeing less important words
        vectorizer = StemmedTfidfVectorizer(min_df=1,
                                            stop_words='english', decode_error='ignore')
        X_train = vectorizer.fit_transform(posts)
        new_post = 'imaging databases'
        new_post_vec = vectorizer.transform([new_post])

        best_dist = sys.maxsize
        best_i = None
        for i, post in enumerate(posts):
            post_vec = X_train.getrow(i)
            d = dist_norm(post_vec, new_post_vec)
            print('== Post {0:d} with dist={1:.2f}: {2:s}'.format(i, d, post))
            if best_dist > d:
                best_dist = d
                best_i = i
        print('Best post is {0:d} with dist={1:.2f}'.format(best_i, best_dist))

    def test_kmeans(self):
        """
        20newsgroups for clustering. We use categories to limit the dataset size.
        """
        groups = ['comp.graphics', 'comp.os.ms-windows.misc',
                  'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                  'comp.windows.x', 'sci.space']

        train_data = sklearn.datasets.fetch_20newsgroups(
            subset='train', categories=groups)
        test_data = sklearn.datasets.fetch_20newsgroups(
            subset='test', categories=groups)

        vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                                            stop_words='english', decode_error='ignore')
        vectorized = vectorizer.fit_transform(train_data.data)
        num_samples, num_features = vectorized.shape
        print('#samples: {0:d}, #features: {1:d}'.format(
            num_samples, num_features))

        # Clustering fit
        num_clusters = 50
        km = KMeans(n_clusters=num_clusters,
                    init='random', n_init=1, verbose=1, random_state=3)
        km.fit(vectorized)

        # Clustering transform
        new_post = """Disk drive problems. Hi, I have a problem with my hard disk.
        After 1 year it is working only sporadically now.
        I tried to format it, but now it doesn't boot any more.
        Any idea? Thanks."""
        new_post_vec = vectorizer.transform([new_post])
        new_post_label = km.predict(new_post_vec)[0]

        # km.labels_ , new_post_label is the same dimension.
        similar_indices = (km.labels_ == new_post_label).nonzero()[0]
        similar = []
        for i in similar_indices:
            dist = dist_raw(new_post_vec, vectorized[i].toarray())
            similar.append((dist, train_data.data[i]))
        similar = sorted(similar)
        print('Similar position {0:d}, similarity {1:.3f}, content {2:s}'.format(
            1, similar[0][0], similar[0][1]))


if __name__ == '__main__':
    unittest.main()
