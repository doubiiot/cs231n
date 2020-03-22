from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        score_i = np.dot(X[i],W)
        #Dividing large numbers can be numerically unstable, so it is important to use a normalization trick. 
        score_i -= np.max(score_i) 
        score_yi = score_i[y[i]]
        row_sum = np.sum(np.exp(score_i))
        loss_i = np.log(row_sum) - score_yi
        loss += loss_i
        for j in range(num_classes):
            dW[:,j] += np.exp(score_i[j]) * X[i] / row_sum
            if j == y[i]:
                dW[:,j] -= X[i]
    loss /= num_train
    loss += reg * np.sum(W * W) # + reg * W^2
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    score = np.dot(X,W)
    #score -= np.amax(score,axis=1) 
    score_y = score[np.arange(num_train), y] # n * 1
    row_sum = np.sum(np.exp(score),axis=1)# n * 1
    loss = np.sum(np.log(row_sum) - score_y)
    
    temp = np.exp(score) / row_sum.reshape(-1,1)
    temp[np.arange(num_train), y] -= 1
    dW = np.dot(X.T,temp)
    
    loss /= num_train
    loss += reg * np.sum(W * W) # + reg * W^2
    dW /= num_train
    dW +=  2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
