import numpy as np
import matplotlib.pyplot as plt


def create_projection(y):
    """
    Creates a projection matrix onto the hyperplane with normal vector y.

    Parameters
    ----------
    y : ndarray(N-samples, 1)
        Training labels; y[i] = c means that X[i] has label c,
        where 0 <= c < C.

    Returns
    -------
    proj : ndarray(N-samples, N-samples)
        Projection matrix.
    """
    I = np.eye(len(y))
    norm = np.sqrt((y*y).sum(axis=0))
    proj = I - (y.dot(y.T) / norm**2)  # (N, N)

    return proj


def rbf_gram_matrix(X, Y=None, gamma=1e-3):
    """
    A vectorized implementation of the Gaussian Radial Basis Function (RBF)
    Kernel grammian, where K(x, y) = exp(-gamma * norm(x-y)^2).

    Parameters
    ----------
    X : ndarray(N-samples, D-dimensions)
        Training data; there are N training samples each of dimension D.

    Y : ndarray(M-samples, D-dimensions) (default None)
        Training data; there are M training samples each of dimension D.

    gamma : float (default 1e-1)
        Controls the (reciprocal) std deviation of the gaussian rbf. The larger
        the value, the greater the risk of over-fitting.

    Returns
    -------
    K : ndarray(N-samples, M-samples)
        Grammian K.
    """
    # get a matrix where the (i, j)th element is |x[i] - x[j]|^2
    # using the identity (x - y)^T (x - y) = x^T x + y^T y - 2 x^T y
    pt_sq_norms_X = (X ** 2).sum(axis=1)
    if Y is None:
        Y = X
        pt_sq_norms_Y = pt_sq_norms_X
    else:
        pt_sq_norms_Y = (Y ** 2).sum(axis=1)
    K = np.dot(X, Y.T)  # slow already parallel
    K *= -2
    K += pt_sq_norms_X.reshape(-1, 1)
    K += pt_sq_norms_Y

    # turn into an RBF gram matrix
    K *= -gamma
    # slow not parallel
    np.exp(K, K)  # exponentiates in-place

    return K


class NonLinearSVM(object):
    """
    A binary non-linear classifier that uses the dual-problem formulation of
    the soft-margin SVM objective. Instead of minimizing an objective w.r.t.
    weights and biases, we are maximizing an objective w.r.t. Lagrange
    multipliers.

    As opposed to our implementation of LinearSVM, we are not dealing with the
    primal formulation, and we are taking the associated constraints into
    account.

    We are going to achieve non-linear classification by using the "Kernel
    Trick", which is only possible through the dual-problem formulation
    because it exposes a gram-matrix operated on the sample data that can
    be replaced with a higher-order kernel.

    i.e. x.T.dot(x) can be replaced with K(x,x) where K can be any function.

    Here are some popular kernels:

    Linear Kernel
        K(x, y) = x.T.dot(y)
        - Algorithms using this kernel are often equivalent to non-kernel
        algorithms such as standard PCA.

    Polynomial Kernel
        K(x, y) = (x.T.dot(y) + offset)^dim
        - Well suited when training data is normalized.

    Gaussian Kernel
        K(x, y) = exp(-norm(x-y)^2 / (2*sigma^2))
        - Infinite dimensionality.

    Gassian Radial Basis Function (RBF) Kernel
        K(x, y) = exp(-gamma * norm(x-y)^2)
        - Infinite dimensionality.
    """

    def __init__(self):
        self.a = None  # Lagrange multiplers
        self.gamma = None
        self.support_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, C=1.0,
              gamma=1e-3, num_epochs=20, verbose=False):
        """
        Train this non-linear classifier using batch dual coordinate ascent.

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

        C : float (default 1.0)
            Soft-margin penalty.

        num_epochs : integer (default 20)
            Number of epochs for optimization.

        verbose : boolean (default False)
            If True, print progress during optimization.

        Returns
        -------
        gain_history : ndarray(num_epochs,)
            A list containing the average gain at each training iteration.
        """
        num_train, dim = X.shape
        y = y.reshape((-1, 1))  # (N, 1)

        if self.a is None:
            self.a = np.zeros((num_train, 1))

        if self.gamma is None:
            self.gamma = gamma

        proj = create_projection(y)  # (N, N)
        print 'Pre-calculating kernel...'
        K = rbf_gram_matrix(X, gamma=self.gamma)  # (N, N)

        # run batch dual coordinate ascent to optimize a
        gain_history = []
        train_acc_history = []
        val_acc_history = []
        for epoch in xrange(num_epochs):
            # evaluate gain and gradient
            gain, grad = self.gain(self.a, K, y)
            gain_history.append(gain)  # record gain at each iteration

            # perform parameter update
            self.a += learning_rate * grad

            # apply inequality constraint: 0 <= a_i <= C
            self.a[self.a < 0] = 0
            self.a[self.a > C] = C

            # apply equality constraint: project onto hyperplane defined by
            # A.T.dot(y_batch) = 0
            self.a = proj.dot(self.a)

            # update support vectors, multipliers, and labels
            self.update_support(X, y)

            # Check accuracy
            train_acc = (np.sign(self.predict(X)) == y).mean()
            val_acc = (np.sign(self.predict(X_val)) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            if verbose:
                num_support = self.support_vectors.shape[0]
                print ('epoch (%d / %d) gain: %f train acc: %f '
                       'val acc: %f lr: %f C: %f gamma: %f '
                       'support vectors: %d/%d') % (epoch, num_epochs,
                                                    gain, train_acc, val_acc,
                                                    learning_rate, C, gamma,
                                                    num_support, num_train)
                # plot distribution of gradients
                myarr = grad.ravel()
                weights = np.ones_like(myarr)/float(len(myarr))
                plt.hist(myarr, 50, normed=False, facecolor='green',
                         weights=weights, alpha=0.75)
                plt.xlabel('Gradients')
                plt.ylabel('Probability')
                plt.title('Histogram of Gradients at Epoch %d' % epoch)
                plt.grid(True)
                plt.show()

        return {
            'gain_history': gain_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained multipliers of this non-linear classifier to predict
        labels for data points.

        Parameters
        ----------
        X : ndarray(N-samples, D-dimensions)
            Training data. Each row has D dimensions.

        Returns
        -------
        y_pred : ndarray(N-samples,)
            Predicted labels for the data in X.
        """
        mul = self.support_multipliers * self.support_vector_labels
        K = rbf_gram_matrix(self.support_vectors, X, gamma=self.gamma)
        wX = np.sum(mul * K, axis=0).reshape(-1, 1)
        K = rbf_gram_matrix(self.support_vectors, gamma=self.gamma)  # slow
        # slow
        b = np.mean(self.support_vector_labels -
                    np.sum(mul * K, axis=0).reshape(-1, 1))

        y_pred = wX + b

        return y_pred

    def gain(self, a, K, y):
        """
        Compute the gain function and its derivative.

        Parameters
        ----------
        a : ndarray(N-samples, 1)
            Lagrange multipliers.

        K : ndarray(N-samples, N-samples)
            Pre-computed RBF grammian.

        y : ndarray(N-samples, 1)
            Labels for the batch.

        Returns
        -------
        gain, da : Tuple(float, ndarray(N-samples, 1))
            Gain and gradient with respect to multipliers a.
        """
        # initialize the gain and gradient to zero
        gain = 0.0
        da = np.zeros(a.shape)

        # compute the objective/gain
        S = (y * K) * y.T

        # L = sum(A) - 0.5 * sum_ij(Ai * Aj * yi * yj * xi.dot(xj))
        # Note: sum(A) = a.T.dot(np.ones(len(a)))
        gain = (np.sum(a) - 0.5 * (a.T.dot(S).dot(a))).item()

        # compute the gradient
        da = np.ones((len(a), 1)) - S.dot(a)  # (N, 1)

        return gain, da

    def update_support(self, X, y):
        """
        Update support vectors, support multipliers, and support vector labels.

        Support vectors and support vector labels correspond to non-zero
        lagrange multipliers. These multipliers are called the support
        multipliers and their corresponding support vectors lie on the margin.

        Parameters
        ----------
        X : ndarray(N-samples, D-dimensions)
            Training data. Each row has D dimensions.

        y : ndarray(N-samples, 1)
            Training labels; y[i] = c means that X[i] has label c,
            where 0 <= c < C.
        """
        support_vector_indices = np.where(self.a > 1e-5)[0]
        self.support_multipliers = self.a[support_vector_indices]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices].reshape((-1, 1))
