import numpy as np


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, reg=1e-5,
              num_iters=100, batch_size=200, verbose=False):
        """
        Train this linear classifier using mini-batch stochastic gradient
        descent.

        Parameters
        ----------
        X : ndarray(N-samples, D-dimensions)
            Training data; there are N training samples each of dimension D.

        y : ndarray(N-samples,)
            Training labels; y[i] = c means that X[i] has label c,
            where 0 <= c < C.

        learning_rate : float (default 1e-3)
            Learning rate for optimization.

        reg : float (default 1e-5)
            Regularization strength.

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
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assuming y labels are 0..K

        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # run mb stochastic gradient descent to optimize W
        iterations_per_epoch = max(num_train / batch_size, 1)
        epoch = 0
        total_epochs = num_iters / iterations_per_epoch
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for i in xrange(num_iters):
            X_batch = None
            y_batch = None

            # sampling with replacement
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient
            loss, grad = self.loss(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)  # record loss at each iteration

            # perform parameter update
            self.W += -learning_rate * grad

            # Every epoch, check train and val accuracy
            if i % iterations_per_epoch == 0:
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

                # TODO: decay learning rate?

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Parameters
        ----------
        X : ndarray(N-samples, D-dimensions)
            Training data. Each row has D dimensions.

        Returns
        -------
        y_pred :  ndarray(N-samples,)
            Predicted labels for the data in X.
        """
        y_pred = X.dot(self.W).argmax(axis=1)
        return y_pred

    def loss(self, W, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. Subclasses will override
        this.

        Parameters
        ----------
        W : ndarray(D-dimensions, C-classes)
            Weights.

        X_batch : ndarray(N-samples, D-dimensions)
            A minibatch of N data points.

        y_batch : ndarray(N-samples,)
            Labels for the minibatch.

        reg : float
            Regularization strength.

        Returns
        -------
        loss, dW : Tuple(float, ndarray(D-dimensions, C-classes))
            Loss and gradient with respect to weights W.
        """
        pass
