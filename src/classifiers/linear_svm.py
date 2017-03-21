from src.classifiers import LinearClassifier
import numpy as np


class LinearSVM(LinearClassifier):
    """
    A subclass that uses the Multiclass SVM loss function as defined by
    Weston and Watkins (1999):

    https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es1999-461.pdf

    This version is more powerful than the OVA version of SVM in the sense that
    you can construct multiclass datasets where this version can achieve zero
    data loss, but OVA cannot. We don't have to create separate classifiers as
    well.

    This is different than the Structured SVM and All-vs-All (AVA) approaches.
    AVA is related while Structured SVMs try to maximize the margin between the
    score of the correct class and the score of the highest-scoring incorrect
    runner-up class.

    Also, we are using the (unconstrained) primal formulation of the objective,
    not the dual-problem formulation which involves optimizing w.r.t. Lagrange
    multipliers as opposed to weights and biases.

    Interestingly, OVA SVM can arguably work just as well as argued in this
    paper:

    http://www.jmlr.org/papers/volume5/rifkin04a/rifkin04a.pdf

    """

    def loss_naive(self, W, X, y, reg):
        """
        Multiclass SVM loss function, naive implementation (with loops).

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
        dW = np.zeros(W.shape)  # initialize the gradient as zero

        # compute the loss and the gradient
        num_classes = W.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        margin = 1
        for i in xrange(num_train):
            scores = X[i].dot(W)
            correct_class_score = scores[y[i]]
            num_contributed_to_loss = 0
            for j in xrange(num_classes):
                if j == y[i]:
                    continue
                error = scores[j] - correct_class_score + margin
                if error > 0:
                    loss += error
                    dW[:, j] += X[i]
                    num_contributed_to_loss += 1
            dW[:, y[i]] += -1 * num_contributed_to_loss * X[i]

        # average loss and gradient over all training examples
        loss /= num_train
        dW /= num_train

        # add regularization to the loss and gradient.
        loss += 0.5 * reg * np.sum(W * W)
        dW += reg * W

        return loss, dW

    def loss(self, W, X, y, reg):
        """
        Multiclass SVM loss function, vectorized.

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
        # initialize the loss and gradient to zero
        loss = 0.0
        dW = np.zeros(W.shape)

        num_classes = W.shape[1]
        num_train = X.shape[0]

        # compute the loss
        margin = 1
        scores = X.dot(W)  # (N, C)
        correct_class_scores = scores[np.arange(len(scores)),
                                      y].reshape((num_train, 1))
        error = scores - correct_class_scores + margin
        error[error < 0] = 0  # don't pay attention to points past the margin
        error[np.arange(len(scores)), y] = 0  # y_i is correct class, no error
        loss = np.sum(error)

        # compute the gradient
        num_contributed_to_loss = error.copy()
        # did contribute to loss, set to 1
        num_contributed_to_loss[error > 0] = 1
        # set -1 * sum of num_contributed_to_loss at column y_i
        num_contributed_to_loss[np.arange(len(scores)),
                                y] = -1 * num_contributed_to_loss.sum(axis=1)
        dW = X.T.dot(num_contributed_to_loss)  # (D, C)

        # average loss and gradient over all training examples
        loss /= num_train
        dW /= num_train

        # add regularization to the loss and gradient.
        loss += 0.5 * reg * np.sum(W * W)
        dW += reg * W

        return loss, dW
