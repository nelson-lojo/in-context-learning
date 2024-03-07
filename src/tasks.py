import math

import torch

from models import throw


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "kalman_filter": KalmanFilter,
        "noisy_kalman_filter": NoisyKalmanFilter
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")  
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class KalmanFilter(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(KalmanFilter, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size
        self._dims = KalmanFilter._get_dims(self.n_dims, self.hidden_layer_size)

        if seeds is not None:
            generator = torch.Generator()
            assert len(seeds) == self.b_size

            pool_dict = {
                # for each parameter name ...
                param_name : torch.cat( [ # generate a batch of values ...
                    KalmanFilter._generate_rand_stable(1, *param_dims, generator=generator.manual_seed(seed)) 
                    for seed in seeds                              # ... each sampled from a different seed
                ] )
                for param_name, param_dims in self._dims.items()
            }
        elif pool_dict is None:
            pool_dict = { param_name : KalmanFilter.generate_rand_stable(self.b_size, *param_dims) 
                          for param_name, param_dims in self._dims.items() }

        assert "A" in pool_dict and "B" in pool_dict and "C" in pool_dict and "x_0" in pool_dict
        assert len(pool_dict["A"]) == len(pool_dict["B"]) == len(pool_dict["C"]) == len(pool_dict["x_0"])
        indices = torch.randperm(len(pool_dict["A"]))[:self.b_size]
        self.A = pool_dict["A"][indices]
        self.B = pool_dict["B"][indices]
        self.C = pool_dict["C"][indices]
        self.x_k_1 = pool_dict["x_0"][indices]


    def evaluate(self, u_k):

        A = self.A.to(u_k.device)
        
        B = self.B.to(u_k.device)
        
        C = self.C.to(u_k.device)
        
        x_k_1 = self.x_k_1.to(u_k.device)
        # Renormalize to Linear Regression Scale
        
        x_k_all = torch.zeros(u_k.size()[0], u_k.size()[1], u_k.size()[2], device=u_k.device)
        x_k_all[:, 0, :] = x_k_1[:, 0, :]

        for i in range(u_k.size()[1] - 1):
            x_k_all[:, (i+1):(i + 2), :] = x_k_all[:, i:(i + 1), :] @ A + u_k[:, (i + 1):(i + 2), :] @ B
        
        y_k = C @ torch.transpose(x_k_all, 1, 2)
        y_k = torch.transpose(y_k, 1, 2)

        #BELOW IS THE SCALING
        y_k = (1/math.sqrt(len(y_k[0, :, 0]))) * y_k
        return y_k[:, :, 0]

    @staticmethod
    def _generate_rand_stable(batch_size, rows, cols, generator=None):
        ans = torch.zeros(batch_size, rows, cols)
    
        for i in range(batch_size):
            e_vals = torch.rand(min(rows, cols), generator=generator)
            e_val_signs = torch.rand(min(rows, cols), generator=generator)
            for j in range(len(e_vals)):
                e_vals[j] *= -1 if (e_val_signs[j] < 0.5) else 1
        
            gaus = torch.randn (rows, rows, generator=generator)
            svd = torch.linalg.svd (gaus)   
            orth1 = svd[0]
            orth1 = orth1[:, :min(rows, cols)]

            gaus = torch.randn (cols, cols, generator=generator)
            svd = torch.linalg.svd (gaus)   
            orth2 = svd[2]
            orth2 = orth2[:min(rows, cols), :]

            ans[i, :, :] = torch.matmul(torch.matmul(orth1, torch.diag(e_vals)), orth2)

        return ans

    # Assume that we are instantiating A, B, and C matrices for the following
    # state space model: 
    #
    #           x_k = A * x_(k-1) + B * u_k
    #           y_k = C * x_k
    #
    # The propostition is to project the input signal u_k into a higher dimension 
    #   this means x_k is (hidden_layer_size, 1), u_k is in_dim, and y_k is (out_dim, 1),
    # A -> (hidden_layer_size, hidden_layer_size), B -> (hidden_layer_size, in_dim), C -> (out_dim, hidden_layer_size)
    @staticmethod
    def _get_dims(in_dim, hidden_size, out_dim=1):
        return {
            "A"   : (hidden_size, hidden_size),
            "B"   : (hidden_size,      in_dim),
            "C"   : (    out_dim, hidden_size),
            "x_0" : (hidden_size,     out_dim)
        }

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=100, **kwargs):
        dims = KalmanFilter._get_dims(n_dims, hidden_layer_size)
        return {
            param_name : KalmanFilter._generate_rand_stable(num_tasks, *param_dim)
            for param_name, param_dim in dims.items()
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class NoisyKalmanFilter(KalmanFilter):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(NoisyKalmanFilter, self).__init__(n_dims, batch_size, pool_dict, seeds, scale, hidden_layer_size)

    def evaluate(self, u_k):

        A = self.A.to(u_k.device)
        
        B = self.B.to(u_k.device)
        
        C = self.C.to(u_k.device)
        
        x_k_1 = self.x_k_1.to(u_k.device)
        # Renormalize to Linear Regression Scale

        x_k_all = torch.zeros(u_k.size()[0], u_k.size()[1], u_k.size()[2], device=u_k.device)
        x_k_all[:, 0, :] = x_k_1[:, 0, :]
        for i in range(u_k.size()[1] - 1):
            x_k_all[:, (i+1):(i + 2), :] = x_k_all[:, i:(i + 1), :] @ A + u_k[:, (i + 1):(i + 2), :] @ B + (torch.randn (u_k.size()[0], 1, u_k.size()[2]) / 5)
        
        y_k = C @ torch.transpose(x_k_all, 1, 2)
        y_k = torch.transpose(y_k, 1, 2)

        #ADD OBSERVATION NOISE
        y_k += (torch.randn (y_k.size()[0], y_k.size()[1], y_k.size()[2]) / 6)
        #BELOW IS THE SCALING
        y_k = (1/math.sqrt(len(y_k[0, :, 0]))) * y_k
        return y_k[:, :, 0]
