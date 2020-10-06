import pytest
import numpy as np

from sklearn.datasets import make_regression
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

from gprEmbedding import SubspaceKernel, EmbeddingRBF


class Test_CustomKernel:
    def test_selection_kernel(self):
        X, y = make_regression(n_samples=150, n_features=12)

        train_X, train_y = X[:100], y[:100]
        test_X, test_y = X[100:], y[100:]

        subsp_kerenl1 = SubspaceKernel(RBF(length_scale=[1]*4), ids_to_apply=np.array([0,1,2,3]))

        string_rep = str(subsp_kerenl1)


        subsp_kerenl2 = SubspaceKernel(RBF(length_scale=[1]*8), ids_to_apply=np.array([4,5,6,7,8,9,10,11]))

        full_kernel = RBF(length_scale=[1]*12)

        k, grad = full_kernel(X[:5], eval_gradient=True)

        sub_kernel = subsp_kerenl1 * subsp_kerenl2

        model = GaussianProcessRegressor(kernel=sub_kernel)

        model.fit(train_X, train_y)


        pred = model.predict(test_X)
        pass


        model_full = GaussianProcessRegressor(kernel = full_kernel)

        model_full.fit(train_X, train_y)

        pred_full = model_full.predict(test_X)


        diff = pred - pred_full

        assert(np.all(np.isclose(pred, pred_full)))

        kernel_vals_full = model_full.kernel_.length_scale

        # kernel_vals_partially = model.kernel_.base_kernel.kernel_.length_scale

    def test_understanding_of_rbf_kernel(self):
        rbf = 1**2*RBF(length_scale=[1.5, 1.5, 1.5], length_scale_bounds='fixed')

        gp = GaussianProcessRegressor(rbf)
        X, y = make_regression(n_features=
                               3)

        gp.fit(X, y)

        print()



        dim = 4
        n_samples = 10

        X = np.random.rand(n_samples, dim)*10 - 5

        l = np.random.rand(dim)*10

        kernel = RBF(length_scale=l)

        outkernel, gradient = kernel(X, eval_gradient=True)

        def _test_output(k, m):
            x = X[k, :]
            y = X[m, :]

            x_scaled = x / l
            y_scaled = y / l

            squared_diff = (x_scaled - y_scaled)**2

            expected = np.exp(- 0.5* np.sum(squared_diff))

            actual = outkernel[k, m]

            assert actual == expected

        _test_output(1, 3)
        _test_output(4, 4)
        _test_output(5, 2)


        def _test_gradient(k, m, i):
            # row k, column m, length scale i

            x_k = X[k, i]
            x_m = X[m, i]

            l_i = l[i]

            kernel_val = outkernel[k, m]

            other_val = ((x_k - x_m)**2) / (l_i**3)

            full = kernel_val * other_val # dK / d l_i

            full_log_transformed = full * l_i # dk/ d log(l_i)

            actual = gradient[k, m, i]

            assert np.isclose( full_log_transformed , actual)

            pass


        _test_gradient(1, 3, 0)
        _test_gradient(1, 3, 3)
        _test_gradient(4, 7, 2)
        print()

    def test_EmbeddingRBF(self):
        self._check_EmbeddingRBF()
        self._check_EmbeddingRBF()


    def _check_EmbeddingRBF(self):
        product_ids = [0, 1, 0, 2, 3, 4, 3, 4, 2, 1]
        n_samples = len(product_ids)
        n_products = int(np.max(product_ids)+1)


        assert n_samples == 10
        assert n_products == 5

        X = np.zeros((n_samples, n_products))
        X[np.arange(n_samples), product_ids] = 1

        projection_dim = 7

        kernel = EmbeddingRBF.make4EntityEmbed(n_entities=n_products, embedding_dimension=projection_dim)
        flat_mat = kernel.flat_matrix


        mat = kernel.vec_to_W(flat_mat)

        kernel_vals, gradient = kernel(X, eval_gradient=True)

        def _check_value(i, j):
            id_i = product_ids[i]
            id_j = product_ids[j]

            w_i = mat[:, id_i]
            w_j = mat[:, id_j]

            dist = np.sum((w_i - w_j)**2)

            expected = np.exp(-0.5*dist)

            actual = kernel_vals[i, j]

            assert np.isclose(expected , actual)

        for i in range(n_samples):
            for j in range(n_samples):
                _check_value(i, j)


        def _check_gradient(i, j, grad_vec_id):
            computed_gradient_value_all = gradient[i, j, :]

            gradient_mat = kernel.vec_to_W(computed_gradient_value_all)

            gradient_vec = gradient_mat[:, grad_vec_id]

            current_W_mat = kernel.W

            current_target_W = current_W_mat[:, grad_vec_id]

            kernel_value = kernel_vals[i, j]

            id_i = product_ids[i]
            id_j = product_ids[j]

            if id_i == id_j:
                assert np.all(gradient_vec == 0)
            elif (id_i != grad_vec_id) and ( id_j != grad_vec_id):
                assert np.all(gradient_vec == 0)

            elif (id_i == grad_vec_id) and (id_j != grad_vec_id):

                w_j = current_W_mat[:, id_j]

                expected_dK_dWtarget = kernel_value * (w_j - current_target_W)

                expected = expected_dK_dWtarget * current_target_W

                assert len(expected) == projection_dim

                assert np.all(np.isclose(expected, gradient_vec))

            elif (id_i != grad_vec_id) and (id_j == grad_vec_id):
                w_i = current_W_mat[:, id_i]

                expected_dK_dWtarget = kernel_value * (w_i - current_target_W)

                expected = expected_dK_dWtarget * current_target_W

                assert len(expected) == projection_dim

                assert np.all(expected == gradient_vec)
            else:
                raise RuntimeError()

        for i in range(n_samples):
            for j in range(n_samples):
                for t in range(n_products): # number of embedding vectors
                    _check_gradient(i, j, t)

    def test_EmbeddingRBF_helper(self):
        self._check_EmbeddingRBF_helper()
        self._check_EmbeddingRBF_helper()

    def _check_EmbeddingRBF_helper(self):
        n_samples_x = 10

        n_samples_y = 11

        n_one_hot = 5
        projection_dim = 7


        # testing conversions from flat vectors to W matrices
        k = EmbeddingRBF.make4EntityEmbed(n_entities=n_one_hot, embedding_dimension=projection_dim)

        flat_mat = k.flat_matrix

        mat_form = k.vec_to_W(flat_mat)
        flat_form = k.W_to_vec(mat_form)
        assert np.all(flat_form == flat_mat)


        # this checks that the embeddings vectors are stored in the flat_mat vector one by one
        assert np.all(mat_form[:, 0] == flat_mat[:projection_dim])


        # testing reshaping of stacked
        stacked_W = np.random.rand(n_samples_x, n_samples_y, projection_dim, n_one_hot)

        unstacked_W = k.stacked_W_to_stacked_vec(stacked_W)

        def _check_unstack(i, j):
            mat = stacked_W[i, j]
            expected = k.W_to_vec(mat)

            actual = unstacked_W[i, j, :]

            assert np.all(actual == expected)

        _check_unstack(1,3)
        _check_unstack(0,5)
        _check_unstack(1,2)
        _check_unstack(5,4)

        W = k.W
        assert np.all(np.isclose(W , mat_form))

        # testing to compute the difference vector for all pairs

        X = np.random.rand(n_samples_x, n_one_hot)
        Y = np.random.rand(n_samples_y, n_one_hot)
        diff = k.vector_differences(X, Y)

        assert diff.shape == (n_samples_x, n_samples_y, n_one_hot)
        def _check_diff(i, j):
            x_i = X[i, :]
            y_j = Y[j, :]

            expected = x_i - y_j

            assert np.all(diff[i, j, :] == expected)

        _check_diff(2, 3)
        _check_diff(4, 2)
        _check_diff(0, 0)

        ############

        # testing the Matrix vector product for all difference pairs
        mat_mutliplied = k.vector_matmul(W, diff)
        assert mat_mutliplied.shape == (n_samples_x, n_samples_y, projection_dim)

        def _check_vector_matmul(i, j):

            diff_vec = diff[i, j]

            expected = W @ diff_vec

            actual = mat_mutliplied[i, j, :]

            assert np.any(actual == expected)


        _check_vector_matmul(0, 3)
        _check_vector_matmul(1, 3)
        _check_vector_matmul(5, 2)
        _check_vector_matmul(3, 3)


        # testing the outer v @ v.T vector->matrix product for all pairs
        outer_vector_product = k.outer_vector_product(diff)

        assert outer_vector_product.shape == (n_samples_x, n_samples_y, n_one_hot, n_one_hot)

        def _check_outer_vector_product(i, j):
            vec = diff[i, j]

            expected = vec[:, None] @ vec[None, :] # vec @ vec.T -> matrix
            actual = outer_vector_product[i, j, :, :]

            assert np.all(expected == actual)

            for ii in range(len(vec)):
                for jj in range(len(vec)):
                    assert actual[ii, jj] == (vec[ii] * vec[jj])



        _check_outer_vector_product(0, 3)
        _check_outer_vector_product(1, 3)
        _check_outer_vector_product(5, 2)
        _check_outer_vector_product(3, 3)


        # testing the matmul for all pairs
        W_matmul_outer = k.matmul(W, outer_vector_product)

        assert W_matmul_outer.shape == (n_samples_x, n_samples_y, projection_dim, n_one_hot)

        def _check_W_matmul(i, j):
            outer_mat = outer_vector_product[i, j, :, :]

            expected = W @ outer_mat
            actual = W_matmul_outer[i, j, :, :]

            assert np.all(expected == actual)

        _check_W_matmul(0, 2)
        _check_W_matmul(1, 2)
        _check_W_matmul(8, 7)
        _check_W_matmul(0, 5)
        _check_W_matmul(7, 3)
        _check_W_matmul(3, 2)


    def _to_one_hot(self, array):

        n_items = int(np.max(array)) + 1

        out = np.zeros((len(array), n_items))

        out[np.arange(len(array)), array] = 1
        return out


    def test_product_embedding_kernel_functionally(self):
        self._check_product_embedding_kernel_functionally()


    def _check_product_embedding_kernel_functionally(self):
        n_products = 12
        embedding_dim=2

        product_ids = np.random.randint(0, n_products, 500)

        X = self._to_one_hot(product_ids)

        n_clusters = 4
        cluster_values = np.random.rand(4)*10 - 5

        target_cluster = product_ids % n_clusters
        y = cluster_values[target_cluster]

        kernel = EmbeddingRBF.make4EntityEmbed(n_entities=n_products, embedding_dimension=embedding_dim
                                               )

        regression = GaussianProcessRegressor(kernel=kernel)

        n_train=250

        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]

        regression.fit(X_train, y_train)

        predicted = regression.predict(X_test)


        diff = predicted - y_test

        kernel_trained = regression.kernel_

        assert np.all(np.isclose(predicted, y_test))
