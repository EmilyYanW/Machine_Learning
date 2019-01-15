#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from q3_word2vec import *
from q3_sgd import *

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

startTime=time.time()
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) /
       dimVectors, np.zeros((nWords, dimVectors))),
    axis=0)

wordVectors1, costs1  = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
        negSamplingCostAndGradient),
    wordVectors, 0.3, 10000, None, True, PRINT_EVERY=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

wordVectors2, costs2  = sgd(
    lambda vec: word2vec_sgd_wrapper(cbow, tokens, vec, dataset, C,
        negSamplingCostAndGradient),
    wordVectors, 0.3, 5000, None, True, PRINT_EVERY=10)

print "sanity check: cost at convergence should be around or below 10"
print "training took %d seconds" % (time.time() - startTime)

# concatenate the input and output word vectors
wordVectors1 = np.concatenate(
    (wordVectors1[:nWords,:], wordVectors1[nWords:,:]),
    axis=0)
# wordVectors = wordVectors[:nWords,:] + wordVectors[nWords:,:]
wordVectors2 = np.concatenate(
    (wordVectors2[:nWords,:], wordVectors2[nWords:,:]),
    axis=0)

visualizeWords = [
    # "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying", "delicious", "smart", "yesterday","tomorrow",
    "today"]

visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs1 = wordVectors1[visualizeIdx, :]
temp1 = (visualizeVecs1 - np.mean(visualizeVecs1, axis=0))
covariance1 = 1.0 / len(visualizeIdx) * temp1.T.dot(temp1)
U1,S1,V1 = np.linalg.svd(covariance1)
coord1 = temp1.dot(U1[:,0:2])

for i in xrange(len(visualizeWords)):
    plt.text(coord1[i,0], coord1[i,1], visualizeWords[i],
        bbox=dict(facecolor='palevioletred', alpha=0.1))
plt.title('Word2Vec Skipgram', fontsize = 15)
plt.xlim((np.min(coord1[:,0]), np.max(coord1[:,0])))
plt.ylim((np.min(coord1[:,1]), np.max(coord1[:,1])))

plt.savefig('q3_word_vectors_skipgram.png')

# word vector visualization for continuous bag of words
visualizeVecs2 = wordVectors2[visualizeIdx, :]
temp2 = (visualizeVecs2 - np.mean(visualizeVecs2, axis=0))
covariance2 = 1.0 / len(visualizeIdx) * temp2.T.dot(temp2)
U2,S2,V2 = np.linalg.svd(covariance2)
coord2 = temp2.dot(U2[:,0:2])

for i in xrange(len(visualizeWords)):
    plt.text(coord2[i,0], coord2[i,1], visualizeWords[i],
        bbox=dict(facecolor='palevioletred', alpha=0.1))
plt.title('Word2Vec Continuous Bag of Words', fontsize = 15)
plt.xlim((np.min(coord2[:,0]), np.max(coord2[:,0])))
plt.ylim((np.min(coord2[:,1]), np.max(coord2[:,1])))

plt.savefig('q3_word_vectors_cbow.png')

plt.figure(figsize = (8,6))
plt.plot(costs1, label  = "skipgram")
plt.plot(costs2, label = "cbow")
plt.legend(fontsize = 13)
plt.title("Cost and Iterations", fontsize = 15)
plt.savefig('cost_iter.png')
