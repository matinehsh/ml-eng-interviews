import numpy as np
from random import shuffle
#from past.builtins import xrange
import pdb

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  N = X.shape[0]
  C = W.shape[1]
  loss2 = 0.0
  delta = 1.0

  for i in range(N):
    scores = np.dot(X[i,:], W)
    correct_class = y[i]
    correct_class_score = scores[correct_class]

    l = 0
    for j in range(C):
      if j != correct_class:
        margin = scores[j] - correct_class_score + delta
        if margin > 0:
          l += margin
    loss2 += l
  loss2 /= N


  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  # loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, loss2 #dW

def svm_loss_half_vectorized(W, X, y, reg):
  N = X.shape[0]
  C = W.shape[1]
  loss = 0.0
  delta = 1.0

  for i in range(N):
    scores = np.dot(X[i,:], W)
    correct_class = y[i]
    correct_class_score = scores[correct_class]
    margin = np.maximum(0, scores - correct_class_score + delta)
    margin[correct_class] = 0
    loss += np.sum(margin)
  return loss/N


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  delta = 1.0

  #pdb.set_trace()
  scores = np.dot(X, W)
  correct_class_scores = scores[np.arange(N), y][:, np.newaxis]
  margin = np.maximum(0, scores - correct_class_scores + delta)
  margin[np.arange(N), y] = 0
  loss = np.sum(margin) / N


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
