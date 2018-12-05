from tqdm import tqdm
from scipy import integrate
from matplotlib import pyplot as plt
import numpy as np

# original functions
from functions import tda


def make_vector(mat_pd, val_t):
    vec = np.sort(np.maximum(np.minimum(
        val_t - mat_pd[:, 0], mat_pd[:, 1] - val_t), 0.0))[::-1]
    int_k = len(np.where(vec != 0))
    return vec, int_k


def make_function(mat_pd):
    return lambda int_k, val_t: make_vector(mat_pd, val_t)[0][int_k]


def plot_average(list_pd, int_k=1, num_slice=100, range_bd=None, val_y=None,
                 show=True, name_save=None):

    if range_bd is None:
        val_min, val_max = tda.parameter_birth_death_pers(list_pd)[0:2]
        val_el = val_max - val_min
        val_ratio = 0.1
        range_bd = np.array([val_min - val_ratio * val_el,
                             val_max + val_ratio * val_el])
    else:
        pass

    bins = np.linspace(range_bd[0], range_bd[1], num_slice)
    num_pd = len(list_pd)
    plt.figure()
    for k in range(int_k):
        vec = np.zeros(num_slice)
        for i, t in enumerate(bins):
            for j in range(num_pd):
                vec[i] += make_function(list_pd[j])(k, t)
            vec[i] /= num_pd
        plt.plot(bins, vec)
    plt.xlim(range_bd[0], range_bd[1])

    if val_y is None:
        plt.ylim(0, (range_bd[1] - range_bd[0]) / 2)
    else:
        plt.ylim(0, val_y)

    if name_save is not None:
        plt.savefig(name_save)
    else:
        pass

    if show:
        plt.show()
    else:
        pass

    plt.close()
    return range_bd, val_y


class Kernel:
    def __init__(self, list_pd, name_rkhs="Linear", range_bd=None,
                 tqdm_bar=False):
        self.__list_pd = list_pd
        self.num_pd = len(list_pd)
        self.name_rkhs = name_rkhs
        self.tqdm_bar = tqdm_bar

        if range_bd is None:
            val_min, val_max = tda.parameter_birth_death_pers(list_pd)[0:2]
            self.range_bd = np.array([val_min, val_max])
        else:
            self.range_bd = range_bd
        self.val_min, self.val_max = self.range_bd

        self.val_tau = None
        self.mat_distance = None

    def __kernel_linear(self, mat_pd_1, mat_pd_2):
        def __inner_product(t):
            vec_1, int_k_1 = make_vector(mat_pd_1, t)
            vec_2, int_k_2 = make_vector(mat_pd_2, t)
            int_k = np.maximum(int_k_1, int_k_2)
            return np.dot(vec_1[0:int_k], vec_2[0:int_k])

        return integrate.quad(__inner_product, self.val_min, self.val_max)[0]

    def __matrix_linear(self):
        __mat_linear = np.empty((self.num_pd, self.num_pd))
        if self.tqdm_bar:
            process_bar = tqdm(total=int(self.num_pd * (self.num_pd + 1) / 2))
            for i in range(self.num_pd):
                for j in range(i + 1):
                    process_bar.set_description("landscape: (%s, %s)" % (i, j))
                    __mat_linear[i, j] = self.__kernel_linear(
                        self.__list_pd[i], self.__list_pd[j])
                    __mat_linear[j, i] = __mat_linear[i, j]
                    process_bar.update(1)
            process_bar.close()
        else:
            for i in range(self.num_pd):
                for j in range(i + 1):
                    __mat_linear[i, j] = self.__kernel_linear(
                        self.__list_pd[i], self.__list_pd[j])
                    __mat_linear[j, i] = __mat_linear[i, j]
        return __mat_linear

    def gram(self):
        mat_linear = self.__matrix_linear()
        mat_gram, self.mat_distance, self.val_tau = tda.make_mat_gram(
            mat_linear, self.name_rkhs)
        return mat_gram

    def kernel(self, mat_pd_1, mat_pd_2, name_rkhs=None):
        if name_rkhs is None:
            name_rkhs = self.name_rkhs
        else:
            pass

        val_c = self.__kernel_linear(mat_pd_1, mat_pd_2)
        if name_rkhs == "Gaussian":
            val_a = self.__kernel_linear(mat_pd_1, mat_pd_1)
            val_b = self.__kernel_linear(mat_pd_2, mat_pd_2)
            return np.exp(-(val_a + val_b - 2.0 * val_c) / (2.0 * self.val_tau))
        else:
            return val_c
