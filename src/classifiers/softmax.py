from src.classifiers import LinearClassifier
import numpy as np


class Softmax(LinearClassifier):
    """
    A Softmax (multinomial logistic regression) classifier using a
    cross-entropy loss function
    """

    def loss_naive(W, X, y, reg):
        """
        Softmax loss function, naive implementation (with loops)

        Parameters
        ----------
        W : ndarray(D-dimensions, C-classes)
            Weights.

        X : ndarray(N-samples, D-dimensions)
            Minibatch of data.

        y : ndarray(N-samples,)
            Training labels; y[i] = c means that X[i] has label c,
            where 0 <= c < C.

        reg : float
            Regularization strength.

        Returns
        -------
        loss, dW : Tuple(float, ndarray(D-dimensions, C-classes))
            Loss and gradient with respect to weights W.
        """
        # initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(W)

        num_classes = W.shape[1]
        num_train = X.shape[0]
        for i in xrange(num_train):
            scores = X[i].dot(W)

            # normalization to prevent numeric instability
            scores -= np.max(scores)

            scores_exp = np.exp(scores)
            softmax_probs = scores_exp / np.sum(scores_exp)
            target_probs = np.zeros(scores.shape)
            target_probs[y[i]] = 1

            for j in xrange(num_classes):
                # cross-entropy loss
                loss += -target_probs[j] * np.log(softmax_probs[j])
                # cross-entropy derivative
                dW[:, j] -= (target_probs[j] - softmax_probs[j]) * X[i]

        # average loss and gradient over all training examples
        loss /= num_train
        dW /= num_train

        # add regularization to the loss and gradient.
        loss += 0.5 * reg * np.sum(W * W)
        dW += reg * W

        return loss, dW

    def loss(self, W, X, y, reg):
        """
        Softmax loss function, vectorized.

        Parameters
        ----------
        W : ndarray(D-dimensions, C-classes)
            Weights.

        X : ndarray(N-samples, D-dimensions)
            Minibatch of data.

        y : ndarray(N-samples,)
            Training labels; y[i] = c means that X[i] has label c,
            where 0 <= c < C.

        reg : float
            Regularization strength.

        Returns
        -------
        loss, dW : Tuple(float, ndarray(D-dimensions, C-classes))
            Loss and gradient with respect to weights W.
        """
        # initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(W)

        num_train = X.shape[0]

        scores = X.dot(W)  # (N, C)
        scores -= scores.max()  # shift to prevent numerical instability
        scores_exp = np.exp(scores)
        sums_exp = np.sum(scores_exp, axis=1)  # sum along the classes

        # softmax activation (N, C)
        softmax_probs = scores_exp / np.vstack(sums_exp)

        # we're going to make a one-hot encoding of the target scores (target
        # labels) for each sample e.g. label '3' = 0001000000
        target_probs = np.zeros_like(scores)  # (N, C)
        target_probs[np.arange(len(target_probs)), y] = 1

        # cross-entropy loss (sum along the classes, then along the samples)
        # output is (N,)
        cross_entropy = np.sum(-target_probs * np.log(softmax_probs), axis=1)
        loss = np.sum(cross_entropy)  # (1,)

        # cross-entropy derivative
        dW = -X.T.dot(target_probs - softmax_probs)  # (D, C)

        # average loss and gradient over all training examples
        loss /= num_train
        dW /= num_train

        # add regularization to the loss and gradient.
        loss += 0.5 * reg * np.sum(W * W)
        dW += reg * W

        return loss, dW
