import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Get dimensions
  num_classes = W.shape[1]
  num_train = X.shape[0]
 
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = np.zeros((num_train, num_classes))

  for i in  range(num_train):
    # Loss function
    f_i = np.dot(X[i,:],W)
    f_yi = np.exp(np.dot(X[i,:],W[:,y[i]]))
    sum_f = np.sum(np.exp(f_i))
    loss += -np.log(f_yi/sum_f)

    # Gradient

    for j in range(num_classes):
      prob_y_equal_k = np.exp(np.dot(X[i,:],W[:,j])) / sum_f
      if j == y[i]:
        dW[:,j] -= X[i,:]*(1-(prob_y_equal_k))
      else:
        dW[:,j] -= X[i,:]*(-(prob_y_equal_k))

  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W**2)
  dW += reg*(W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  score = np.dot(X,W)
  e_score = np.exp(score)

  sum_lines = np.sum(e_score, axis=1)
  y_e_values = e_score[range(num_train),y]

  loss = np.sum(-np.log(y_e_values / sum_lines))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)

  prob_matrix = e_score / sum_lines[np.newaxis].T
  prob_matrix[range(num_train), y] = -(1 - e_score[range(num_train), y]/sum_lines)

  dW = np.dot(X.T, prob_matrix)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

