from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin, NormalizedKernelMixin, _num_samples, \
    Hyperparameter

import numpy as np

class SubspaceKernel(Kernel):

    def __init__(self, base_kernel, ids_to_apply):
        """
        This is used to apply a kernel only to a subset of all available feature columns.
        I.e. it allows to apply different kernels to different features.
        e.g. RBF kernel to 'concentration features', Embedding Kernel to categorical features.

        :param base_kernel: The kernel to apply
        :param ids_to_apply: (list/nparray of integers) the indices tto apply the kernel to
        """
        if not isinstance(base_kernel, Kernel):
            raise ValueError("base_kernel has to be a {} instance".format(Kernel))

        self.base_kernel = base_kernel
        self.ids_to_apply = ids_to_apply

    def _subindex(self, X):
        if X is None:
            return None

        new = X[:, self.ids_to_apply]

        return new

    def __repr__(self):
        sub_kernel_str = self.base_kernel.__repr__()

        return "SubspaceKernel({} on {})".format(sub_kernel_str, self.ids_to_apply)

    def __call__(self, X, Y=None, eval_gradient=False):
        nX = self._subindex(X)
        nY = self._subindex(Y)

        return self.base_kernel.__call__(X=nX, Y=nY, eval_gradient=eval_gradient)

    def diag(self, X):
        nX = self._subindex(X)
        return self.base_kernel.diag(nX)

    def is_stationary(self):
        return self.base_kernel.is_stationary

    @property
    def requires_vector_input(self):
        return True

    @property
    def n_dims(self):
        return len(self.ids_to_apply)

    @property
    def hyperparameters(self):
        return self.base_kernel.hyperparameters

    @property
    def theta(self):
        return self.base_kernel.theta

    @theta.setter
    def theta(self, theta):
        self.base_kernel.theta = theta

    @property
    def bounds(self):
        return self.base_kernel.bounds


class EmbeddingRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    @classmethod
    def make4EntityEmbed(cls, n_entities, embedding_dimension, flat_matrix_bounds=(0.1, 5.0)
                         ):
        """
        Helper method to initialise an EmbeddingRBF kernel for use as an embedding
        Kernel for categorical data. It allows sensible
        initialisation of the parameters.
        The input data to this kernel has to be one-hot representation of the categorical id


        :param n_entities: number of discrete entities/products/customers (one embedding for each)
        :param embedding_dimension: the length of each embedding vector
        :param flat_matrix_bounds: 2-tuple, bounds of the hypercube that the embeddings live in
        :return: a kernel instance
        """


        mmin, mmax = flat_matrix_bounds

        assert type(embedding_dimension) == int
        assert type(n_entities) == int

        flat_matrix = np.random.rand(embedding_dimension * n_entities) * (mmax - mmin) + mmin





        return cls(flat_matrix=flat_matrix,
                   projection_dimensionality=embedding_dimension,
                   flat_matrix_bounds=flat_matrix_bounds,
                   check_input_onehot=True
                   )


    def __init__(self, flat_matrix, projection_dimensionality, flat_matrix_bounds,
                 check_input_onehot=True):
        """
        If you want to use this for learned embeddiings of categorical data see the alternative constructor
            `MatrixRBF.make4embed`

        A general Kernel that implements RBF(Wx, Wy)
            - first a projection of the data vector by multiplying with a matrix
            - application of an RBF kernel with fixed length_scale=1

        :param flat_matrix: the flat matrix W (i.e. mat.reshape(-1, order='F'))
        :param projection_dimensionality: integer (W.shape[0]), i.e. the projection dimension. this is needed to reshape the flat vector back into the appropriate matrix
        :param flat_matrix_bounds: bounds on the parameters
        :param check_input_onehot: boolean (if True an error is thrown if x/y are not onne hot vectors, usefull if used for embedding of categorical data)
        """
        assert len(flat_matrix.shape) == 1

        self.projection_dimensionality = projection_dimensionality
        self.input_dim = int(len(flat_matrix) / self.projection_dimensionality)

        self.flat_matrix = flat_matrix
        self.flat_matrix_bounds = flat_matrix_bounds

        self.check_input_onehot = check_input_onehot


    @property
    def hyperparameter_flat_matrix(self):
        return Hyperparameter(
            "flat_matrix", "numeric", self.flat_matrix_bounds, len(self.flat_matrix))

    def vec_to_W(self, vec):
        return vec.reshape(self.projection_dimensionality, -1, order='F') #  projection_dim, input_dim

    def W_to_vec(self, W):
        return W.reshape(-1, order='F')

    def stacked_W_to_stacked_vec(self, W_stack):
        """
        reshapes stacked W in a way that is consistent with the above to methods

        :param W_stack: n_x, n_y, projection_dim, input_dim
        :return: unstacked: n_x, n_y, projection_dim * input_dim
        """
        assert W_stack.shape[2] == self.projection_dimensionality
        assert W_stack.shape[3] == self.input_dim

        n_x = W_stack.shape[0]
        n_y = W_stack.shape[1]

        return W_stack.reshape(n_x, n_y, -1, order='F')


    @property
    def W(self):

        vec = self.flat_matrix

        return self.vec_to_W(vec)


    def _check_input(self, inp):
        n_samples, n_dim = inp.shape

        assert n_dim == self.W.shape[1]

        if self.check_input_onehot:
            mask = (inp == 1.0) | (inp == 0.0)
            if not np.all(mask):
                raise ValueError(' check_input_onehot=True was given in __init__ but the input data has values that are not 0 or 1')


    def __call__(self, X, Y=None, eval_gradient=False):
        self._check_input(X)

        if Y is None:
            yy = X
        else:
            self._check_input(Y)
            yy = Y

        W = self.W

        diff = self.vector_differences(X, yy) #n_x, n_y, n_dim
        multiplied = self.vector_matmul(W, diff)
        squared = np.sum(multiplied**2, axis=2) # n_x, n_y
        kernel_value = np.exp(-0.5 * squared)

        if eval_gradient:
            assert Y is None

            outer_xy = self.outer_vector_product(diff) # n_x, n_x, n_dim, n_dim
            extra_term = - self.matmul(W, outer_xy)

            full_dK_dW = kernel_value[:, :, None, None] * extra_term # n_x, n_x, projection_dim, input_dim

            # but we need the derivative with respect to log() of parameters -> so multiply by current values
            full_dk_dlogW = full_dK_dW * W[None, None, :, :]


            # now we actually have to flatten it though
            flatt_gradient = self.stacked_W_to_stacked_vec(full_dk_dlogW)

            return kernel_value, flatt_gradient


        else:
            return kernel_value

    @classmethod
    def vector_differences(self, X, Y):
        """

        :param X: martix n_samples_x, n_dim
        :param Y: n_samples_y, n_dim
        :return:  n_samples_x, n_samples_y, n_dim where result[i, j, :] is the vector X[i, :] - Y[j, :]
        """
        return X[:, np.newaxis, :] - Y[np.newaxis, :, :]

    @classmethod
    def vector_matmul(cls, W, X):
        """

        :param W: projection_dim, input_dim
        :param X: n_1, n_2, input_dim
        :return: n_1, n_2, projection_dim, where result[i, j, :] is the vector obtained from W @ X[i, j]
        """

        x_blowup = X[:, :, :, None] # represent the vectors as a input_dim x 1 matrix (i.e. column vector
        out = np.matmul(W, x_blowup)

        assert out.shape[3] == 1
        return out[:, :, :, 0]



    @classmethod
    def outer_vector_product(cls, a):
        """
        :param a: n_1, n_2, n_dim
        :return: n_1, n_2, n_dim, n_dim where result[i, j, : , :] is the matrix
            resulting from the vector to matrix product a[i, j, :] @ a[i, j, :].T
        """

        blown_up_a_left = a[:, :, :, None] # now a's as column vector in the last 2 dimensions
        blown_up_aT_right = a[:, :, None, :] # a^T as row vector in the last 2 dimensions

        multiplied = np.matmul(blown_up_a_left, blown_up_aT_right)
        return multiplied

    @classmethod
    def matmul(cls, W, right):
        """

        :param W: projection_dim, input_dim
        :param right: n_x, n_x, input_dim, input_dim
        :return: n_x, n_x, projection_dim, input_dim -> where result[i, j, :, :] = W @ right[i, j, :, :]
        """

        out = np.matmul(W, right)
        return out