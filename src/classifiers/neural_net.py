import numpy as np


# TODO: learn leak parameter through backprop
def PReLU(X, leak=0.0):
    """
    Compute the Parameterized Rectified Linear Unit, with a given leak
    rate.

    PReLU = max(leak * X, X)

    Parameters
    ----------
    X : ndarray(N-samples, D-dimensions)
        Minibatch of data.

    leak : float (default 0.0)
        Leakiness of PReLU.

    Returns
    -------
    prelu : ndarray(N-samples, D-dimensions)
        Input, rectified.
    """
    prelu = np.maximum(leak * X, X)
    return prelu


def cross_entropy_loss(P, Q):
    """
    Compute the negative log probability (cross-entropy) between
    distributions P and Q.

    Cross-entropy = H(P,Q)
                  = H(P) + DKL(P||Q)
                  = -sum(P * log(P)) + sum(P * log(P/Q))
                  = -sum(P * log(Q))

    Parameters
    ----------
    P : ndarray(N-samples, C-dimensions)
        Expected distribution.

    Q : ndarray(N-samples, C-dimensions)
        Actual distribution.

    Returns
    -------
    loss : float
        Cross-entropy loss between distributions P and Q.
    """
    # cross-entropy loss (sum along the classes, then along the samples)
    # output is (N,)
    cross_entropy = np.sum(-P * np.log(Q), axis=1)
    loss = np.sum(cross_entropy)  # (1,)

    return loss


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network.

    The network has the following architecture:

    input - fully connected layer - PReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Weights are initialized to small random values and biases are
        initialized with zero.

        W1 : First layer weights; has shape (D, H)
        b1 : First layer biases; has shape (H,)
        W2 : Second layer weights; has shape (H, C)
        b2 : Second layer biases; has shape (C,)

        Parameters
        ----------
        input_size : integer
            The dimension D of the input data.

        hidden_size : integer
            The number of neurons H in the hidden layer.

        output_size : integer
            The number of classes C.

        std : float or string (default 1e-4)
            Standard deviation of normally initialized weights and biases.
            Can specify 'xavier' or 'msra' initialization.
        """
        std1 = std
        std2 = std
        if std == "xavier":
            std1 = 1 / np.sqrt((input_size + hidden_size) / 2)
            std2 = 1 / np.sqrt((hidden_size + output_size) / 2)
        elif std == "msra":
            # Kaiming/MSRA weight init
            std1 = 1 / np.sqrt(((input_size + hidden_size) / 2) / 2)
            std2 = 1 / np.sqrt(((hidden_size + output_size) / 2) / 2)

        self.params = {}

        self.params['W1'] = np.random.randn(input_size, hidden_size) * std1
        self.params['b1'] = np.zeros(hidden_size) + std1
        self.params['W2'] = np.random.randn(hidden_size, output_size) * std2
        self.params['b2'] = np.zeros(output_size) + std2

    def train(self, X, y, X_val, y_val, learning_rate=1e-3,
              learning_rate_decay=0.95, momentum=False, reg=1e-5,
              leak=0.01, num_iters=100, batch_size=200, verbose=False):
        """
        Train this neural network using mini-batch stochastic gradient descent.

        Parameters
        ----------
        X : ndarray(N-samples, D-dimensions)
            Training data; there are N training samples each of dimension D.

        y : ndarray(N-samples,)
            Training labels; y[i] = c means that X[i] has label c,
            where 0 <= c < C.

        X_val : ndarray(N_val-samples, D-dimensions)
            Validation data.

        y_val : ndarray(N,)
            Training labels for validation data.

        learning_rate : float (default 1e-3)
            Learning rate for optimization.

        learning_rate_decay : float (default 0.95)
            Scalar factor used to decay the learning rate after each epoch.

        momentum : bool or string (default False)
            Decides which method of momentum to use. True results in vanilla
            momentum update with factor 0.9. 'Nesterov' is another option.

        reg : float (default 1e-5)
            Regularization strength.

        leak : float (default 0.01)
            Leakiness of the PReLU units.

        num_iters : integer (default 100)
            Number of iterations for optimization.

        batch_size : integer (default 200)
            Number of training examples to use at each iteration.

        verbose : boolean (default False)
            If True, print progress during optimization.

        Returns
        -------
        loss_history : ndarray(num_iters,)
            A list containing the average loss at each training iteration.
        """
        num_train = X.shape[0]

        # run mb stochastic gradient descent to optimize W2, b2, W1, and b1
        iterations_per_epoch = max(num_train / batch_size, 1)
        epoch = 0
        total_epochs = num_iters / iterations_per_epoch
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        vW2 = 0
        vb2 = 0
        vW1 = 0
        vb1 = 0
        for i in xrange(num_iters):
            X_batch = None
            y_batch = None

            # sampling with replacement
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient for the current mini-batch
            loss, grads = self.loss(X_batch, y_batch, reg, leak)
            loss_history.append(loss)  # record loss at each iteration

            if momentum is True:
                vW2 = 0.9 * vW2 - learning_rate * grads['W2']
                vb2 = 0.9 * vb2 - learning_rate * grads['b2'].T
                vW1 = 0.9 * vW1 - learning_rate * grads['W1']
                vb1 = 0.9 * vb1 - learning_rate * grads['b1'].T

                # perform parameter updates
                self.params['W2'] += vW2
                self.params['b2'] += vb2
                self.params['W1'] += vW1
                self.params['b1'] += vb1
            elif momentum == 'Nesterov':
                vW2_prev = vW2
                vW2 = 0.9 * vW2 - learning_rate * grads['W2']
                vb2_prev = vb2
                vb2 = 0.9 * vb2 - learning_rate * grads['b2'].T
                vW1_prev = vW1
                vW1 = 0.9 * vW1 - learning_rate * grads['W1']
                vb1_prev = vb1
                vb1 = 0.9 * vb1 - learning_rate * grads['b1'].T

                # perform parameter updates
                self.params['W2'] += -0.9 * vW2_prev + (1 + 0.9) * vW2
                self.params['b2'] += -0.9 * vb2_prev + (1 + 0.9) * vb2
                self.params['W1'] += -0.9 * vW1_prev + (1 + 0.9) * vW1
                self.params['b1'] += -0.9 * vb1_prev + (1 + 0.9) * vb1
            else:
                # perform parameter updates
                self.params['W2'] += -learning_rate * grads['W2']
                self.params['b2'] += -learning_rate * grads['b2'].T
                self.params['W1'] += -learning_rate * grads['W1']
                self.params['b1'] += -learning_rate * grads['b1'].T

            # Every epoch, check train and val accuracy and decay learning rate
            if (i + 1) % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                epoch += 1

                if verbose:
                    print ('epoch (%d / %d): loss: %f train acc: %f '
                           'val acc: %f lr: %f') % (epoch, total_epochs, loss,
                                                    train_acc, val_acc,
                                                    learning_rate)

                if epoch % 2 == 0:
                    # Decay learning rate
                    learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X, leak=0.0):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Parameters
        ----------
        X : ndarray(N-samples, D-dimensions)
            Training data. Each row has D dimensions.

        leak : float (default 0.0)
            Leakiness of PReLU units.

        Returns
        -------
        y_pred :  ndarray(N-samples,)
            Predicted labels for the data in X.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        scores = None
        y_pred = np.zeros(X.shape[1])

        """ FORWARD PASS (w.o. loss) """
        # forward pass up to but not including classification
        # input layer X has shape (N, D)
        fc1 = X.dot(W1) + b1  # fully-connected (fc1) layer (N, H)
        ac1 = PReLU(fc1, leak)  # fc1 activations (ac1) (N, H)
        fc2 = ac1.dot(W2) + b2  # fully-connected (fc2) layer (N, C)

        # compute the softmax scores/predictions
        scores = fc2
        scores -= scores.max()  # shift to prevent numerical instability
        scores_exp = np.exp(scores)
        sums_exp = np.sum(scores_exp, axis=1)  # sum along the classes
        # output layer activations (N, C) i.e. softmax activations
        softmax_probs = scores_exp / np.vstack(sums_exp)
        out = softmax_probs
        """ END OF FORWARD PASS (w.o. loss) """

        y_pred = np.argmax(out, axis=1)

        return y_pred

    def loss(self, X, y, reg=0.0, leak=0.0):
        """
        Compute the loss and gradients for a two-layer fully connected neural
        network.

        Parameters
        ----------
        W : ndarray(D-dimensions, C-classes)
            Weights.

        X : ndarray(N-samples, D-dimensions)
            Minibatch of data.

        y : ndarray(N-samples,)
            Training labels; y[i] = c means that X[i] has label c,
            where 0 <= c < C.

        reg : float (default 0.0)
            Regularization strength.

        leak : float (default 0.0)
            Leakiness of the PReLU units.

        Returns
        -------
        loss, grads : Tuple(float, ndarray(D-dimensions, C-classes))
            Loss and gradient with respect to weights W. grads is a dictionary
            that maps parameter names to the gradients of those parameters
            w.r.t. the loss function.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        """ FORWARD PASS (w. loss) """
        scores = None
        loss = None

        # forward pass up to but not including classification
        # input layer X has shape (N, D)
        fc1 = X.dot(W1) + b1  # fully-connected (fc1) layer (N, H)
        ac1 = PReLU(fc1, leak)  # fc1 activations (ac1) (N, H)
        fc2 = ac1.dot(W2) + b2  # fully-connected (fc2) layer (N, C)

        # compute the softmax scores/predictions
        scores = fc2
        scores -= scores.max()  # shift to prevent numerical instability
        scores_exp = np.exp(scores)
        sums_exp = np.sum(scores_exp, axis=1)  # sum along the classes
        # output layer activations (N, C) i.e. softmax activations
        softmax_probs = scores_exp / np.vstack(sums_exp)
        out = softmax_probs

        # we're going to make a one-hot encoding of the target scores
        # (target labels) for each sample e.g. label '3' = 0001000000
        target_probs = np.zeros_like(out)  # (N, C)
        target_probs[np.arange(len(target_probs)), y] = 1

        # finally, compute the loss L between the output and targets
        loss = cross_entropy_loss(target_probs, out)

        # average loss over all training examples
        loss /= N

        # add regularization to the loss
        loss += 0.5 * reg * np.sum(W1 * W1)
        loss += 0.5 * reg * np.sum(W2 * W2)
        """ END OF FORWARD PASS (w. loss) """

        """ BACKWARD PASS """
        # backprop into fully-connected layer fc2
        # dL/dfc2 = (dL/dout) * (dout/dfc2)
        #         = y_i - target_score => y_i - 1
        dLdfc2 = out
        dLdfc2[np.arange(N), y] -= 1  # (N, C)

        # backprop into params W2 and b2
        # dL/dW2 = (dL/dfc2) * (dfc2/dW2)
        #        = (dL/dfc2) * ac1
        dLdW2 = (ac1.T).dot(dLdfc2)  # (H, C)
        # dL/db2 = (dl/dfc2) * (dfc2/db2)
        #        = (dl/dfc2) * 1 = (dl/dfc2)
        dLdb2 = np.sum(dLdfc2, axis=0)  # sum along samples (C,)

        # backprop into activation layer ac1
        # dL/dac1 = (dL/dfc2) * (dfc2/dac1)
        #         = (dL/dfc2) * W2
        dLdac1 = dLdfc2.dot(W2.T)  # (N, H)

        # backprop into fully-connected layer fc1
        # dL/dfc1 = (dL/dac1) * (dac1/dfc1)
        #         = (dL/dac1) * derivative of PReLU w.r.t. fc1
        # compute derivative of PReLU first
        dac1dfc1 = fc1
        dac1dfc1[dac1dfc1 <= 0] = leak
        dac1dfc1[dac1dfc1 > 0] = 1  # (N, H)
        # finish computing rest of gradient
        dLdfc1 = dLdac1 * dac1dfc1  # (N, H)

        # backprop into params W1 and b1
        # dL/dW1 = (dL/dfc1) * (dfc1/dW1)
        #        = (dL/dfc1) * X
        dLdW1 = (X.T).dot(dLdfc1)  # (D, H)
        # dL/db1 = (dl/dfc1) * (dfc1/db1)
        #        = (dl/dfc1) * 1 = (dl/dfc1)
        dLdb1 = np.sum(dLdfc1, axis=0)  # (H,)

        # average gradients over all training examples
        dLdW2 /= N
        dLdb2 /= N
        dLdW1 /= N
        dLdb1 /= N

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dLdW2 += reg * W2
        dLdW1 += reg * W1

        # gather up all the computed gradients and store in a dict
        grads = {'W1': dLdW1, 'b1': dLdb1,  'W2': dLdW2, 'b2': dLdb2}
        """ END OF BACKWARD PASS """

        return loss, grads
