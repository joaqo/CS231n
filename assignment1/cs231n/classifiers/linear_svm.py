import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
      
        dW[:,y[i]] -= X[i,:] #Change in the correct class improves the score on every class
        dW[:,j] += X[i,:]    #so it is substracted on each loop. It is (num_classes - 1) 
                             #more impactful than a change on one of the other classes.

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  num_classes = W.shape[1]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X,W)
  correct_scores = scores[range(num_train),y][np.newaxis].T

  matrix = np.maximum(0,scores - correct_scores + 1)
  matrix[range(num_train), y] = 0

  loss = np.sum(matrix)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)
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

  filter_matrix = matrix
  filter_matrix[matrix > 0] = 1

  filter_y = np.zeros((num_train, num_classes))
  for index, element in enumerate(y):
      filter_y[index, element] = np.sum(filter_matrix[index,:])

  dW = np.dot(X.T, filter_matrix) - np.dot(X.T, filter_y)
  dW /= num_train
  dW += reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
