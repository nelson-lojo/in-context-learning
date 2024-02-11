import math

import torch


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
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, curriculum=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "kalman_filter": KalmanFilter
    }
    print(curriculum)
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            print(task_name == "kalman_filter")
            if task_name == "kalman_filter":
                print(curriculum)
                pool_dict = KalmanFilter.generate_pool_dict(n_dims, num_tasks, curriculum=curriculum, **kwargs)
                return lambda **args: KalmanFilter(n_dims, batch_size, curriculum, pool_dict=pool_dict, **args, **kwargs)
            else:    
                pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return (lambda **args: KalmanFilter(n_dims, batch_size, curriculum, pool_dict, **args, **kwargs)) if task_name == "kalman_filter" else (lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs))
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
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
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
        # Renormalize to Linear Regression Scale
        print("xs_b size: " + str(xs_b.size()))
        print("W1 size: " + str(W1.size()))
        print("W2 size: " + str(W2.size()))
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)
        print("ys_b_nn size: " + str(ys_b_nn.size()))
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
        curriculum,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(KalmanFilter, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size
        self.curriculum = curriculum
    
        #print(self.curriculum.n_points)
        if pool_dict is None and seeds is None:
            self.A = torch.randn(self.b_size, self.n_dims, self.n_dims)
            self.B = torch.randn(self.b_size, self.n_dims, self.n_dims)
            self.C = torch.randn(self.b_size, 1, self.n_dims)
            self.x_k_1 = torch.randn(self.b_size, 1, self.n_dims)
        elif seeds is not None:
            self.A = torch.randn(self.b_size, self.n_dims, self.n_dims)
            self.B = torch.randn(self.b_size, self.n_dims, self.n_dims)
            self.C = torch.randn(self.b_size, 1, self.n_dims)
            self.x_k_1 = torch.randn(self.b_size, 1, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)

                self.A[i] = torch.randn(self.n_dims, self.n_dims, generator=generator)
                self.B[i] = torch.randn(self.n_dims, self.n_dims, generator=generator)
                self.C[i] = torch.randn(1, self.n_dims, generator=generator)
                self.x_k_1[i] = torch.randn(1, self.n_dims, generator=generator)
        else:
            assert "A" in pool_dict and "B" in pool_dict and "C" in pool_dict and "x_k_1" in pool_dict
            assert len(pool_dict["A"]) == len(pool_dict["B"]) == len(pool_dict["C"]) == len(pool_dict["x_k_1"])
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]
            self.B = pool_dict["B"][indices]
            self.C = pool_dict["C"][indices]
            self.x_k_1 = pool_dict["x_k_1"][indices]


    #Unsure if there's some normalization I must do 
    #How would batch size work in this case?
    def evaluate(self, u_k):
        # I do not know what this does
        #print("A type: " + str(type(self.A)))
        #print("B type: " + str(type(self.B)))
        #print("C type: " + str(type(self.C)))
        #print("x_k_1 type: " + str(type(self.x_k_1)))

        A = self.A.to(u_k.device)
        
        B = self.B.to(u_k.device)
        
        C = self.C.to(u_k.device)
        
        x_k_1 = self.x_k_1.to(u_k.device)
        # Renormalize to Linear Regression Scale

        #print("A type: " + str(type(A)))
        #print(A)
        #print("B type: " + str(type(B)))
        #print(B)
        ##print("C type: " + str(type(C)))
        #print(C)
        ##print("u_k type: " + str(type(u_k)))
        #print(u_k)
        #print("x_k_1 type: " + str(type(x_k_1)))
        #print(x_k_1)
        #print("\n\n\n\n\n\n\n")
       # print("THESE ARE MY INPUTS")
       # print("\n\n\n\n\n\n\n\n")
       # print(u_k)
       # print("\n\n\n\n\n\n\n\n")

        x_k_all = torch.zeros(u_k.size()[0], u_k.size()[1], u_k.size()[2], device=u_k.device)
        x_k_all[:, 0, :] = x_k_1
        #print(x_k_all)
        for i in range(u_k.size()[1] - 1):
            #print(x_k_all[:, i, :].size())
            #print(A.size())
            #print( u_k[:, i + 1, :].size())
            #print(B.size())
            #print((x_k_all[:, i, :] @ A).size())
            #print((u_k[:, i + 1, :] @ B).size())
            x_k_all[:, i+1, :] = x_k_all[:, i, :] @ A + u_k[:, i + 1, :] @ B
        
        #print("\n\n\n\n\n\n\n\n")
        #print(x_k_all)
        ##print("AFTER FOR LOOP")
        #print(x_k_all.size())
        #print(C.size())
        y_k = C @ torch.transpose(x_k_all, 1, 2)
        y_k = torch.transpose(y_k, 1, 2)
        #print(u_k.size())
        #print((A @ x_k_1).size())
        #print((B @ u_k).size())
        #x_k_1 = A @ x_k_1 + B @ u_k
        #y_k =  C @ x_k_1
        #print(B.size())
        #print(C.size())
        #print("y_k type: " + str(y_k.size()))
        #FIXME: NOT SURE IF THIS IS THE RIGHT SPLICING TO DO!!!!!!!

        #FIXME: is this sort of normalization even necessary????
        #y_k = y_k * math.sqrt(2 / self.hidden_layer_size)

        #FIXME: UNCOMMENT THE SCALING
        #y_k = self.scale * y_k
        return y_k

    # Assume that we are instantiating A, B, and C matrices for the following
    # state space model: 
    #
    #           x_k = A * x_(k-1) + B * u_k
    #           y_k = C * x_k
    #
    # If x_k is (n_dims, 1), u_k is scalar, and y_k is (hidden_layer_size, 1),
    # A -> (n_dims, n_dims), B -> (n_dims, 1), C -> (hidden_layer_size, n_dims)
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, curriculum=None, hidden_layer_size=100, **kwargs):
        return {
            "A": torch.randn(num_tasks, n_dims, n_dims),
            "B": torch.randn(num_tasks, n_dims, n_dims),
            "C": torch.randn(num_tasks, 1, n_dims),
            "x_k_1": torch.randn(num_tasks, 1, n_dims)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error