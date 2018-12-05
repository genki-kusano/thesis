from tqdm import tqdm
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np

# original functions
from functions import tda


def plot(mat_pi, val_y=None, range_bd=None, diag=True, name_save=None,
         show=False):
    num_mesh = mat_pi.shape[0]
    plt.figure()
    plt.imshow(np.transpose(mat_pi), interpolation="nearest",
               origin='lower', cmap="YlOrRd")
    plt.colorbar()

    if val_y is None:
        plt.clim(0, mat_pi.max())
    else:
        plt.clim(0, val_y)

    if range_bd is not None:
        plt.xticks([0, num_mesh], [range_bd[0], range_bd[1]])
        plt.yticks([0, num_mesh], [range_bd[0], range_bd[1]])
    else:
        pass

    if diag:
        diagonal = np.linspace(0, num_mesh - 1, 2)
        plt.plot(diagonal, diagonal, "k-", linewidth=0.3)
    else:
        pass

    if name_save is not None:
        plt.savefig(name_save)
    else:
        pass

    if show:
        plt.show()
    else:
        pass
    plt.close()


class Kernel:
    def __init__(self, list_pd, func_weight, val_sigma, num_mesh=80,
                 name_rkhs="Linear", range_bd=None, tqdm_bar=False):
        self.__list_pd = list_pd
        self.num_pd = len(list_pd)
        self.val_sigma = val_sigma
        self.num_mesh = num_mesh
        self.name_rkhs = name_rkhs
        self.tqdm_bar = tqdm_bar

        if range_bd is None:
            val_min, val_max = tda.parameter_birth_death_pers(list_pd)[0:2]
            val_el = val_max - val_min
            val_ratio = 0.1
            self.range_bd = np.array([val_min - val_ratio * val_el,
                                      val_max + val_ratio * val_el])
        else:
            self.range_bd = range_bd
        self.val_min, self.val_max = self.range_bd

        self.val_tau = None
        self.mat_distance = None

        self.val_mesh = (self.val_max - self.val_min) / self.num_mesh
        self.bins = [np.linspace(self.val_min, self.val_max, self.num_mesh + 1),
                     np.linspace(self.val_min, self.val_max, self.num_mesh + 1)]

        self.val_sigma_scaled = self.val_sigma / self.val_mesh
        self.mat_weight = np.zeros((self.num_mesh, self.num_mesh))
        for j in range(self.num_mesh):
            for i in range(j + 1):
                vec_bd = np.array([(i + 0.5) * self.val_mesh + self.val_min,
                                   (j + 0.5) * self.val_mesh + self.val_min])
                self.mat_weight[i, j] = func_weight(vec_bd)

        self.hist = self.__list_hist()
        self.data = self.__list_pi()

    def __list_hist(self):
        list_hist = []
        for k in range(self.num_pd):
            vec_birth = self.__list_pd[k][:, 0]
            vec_death = self.__list_pd[k][:, 1]
            list_hist.append(
                np.histogram2d(vec_birth, vec_death, bins=self.bins)[0])
        return list_hist

    def __list_pi(self):
        list_pi = []
        list_hist = self.hist
        for k in range(self.num_pd):
            list_pi.append(ndimage.filters.gaussian_filter(
                np.multiply(self.mat_weight, list_hist[k]),
                sigma=self.val_sigma_scaled))
        return list_pi

    def __matrix_linear(self):
        mat_pi_vec = np.empty((self.num_pd, self.num_mesh ** 2))
        for k in range(self.num_pd):
            mat_pi_vec[k] = self.data[k].reshape(-1)
        __mat_linear = np.empty((self.num_pd, self.num_pd))
        if self.tqdm_bar:
            process_bar = tqdm(total=int(self.num_pd * (self.num_pd + 1) / 2))
            for i in range(self.num_pd):
                for j in range(i + 1):
                    process_bar.set_description("image: (%s, %s)" % (i, j))
                    __mat_linear[i, j] = np.inner(mat_pi_vec[i], mat_pi_vec[j])
                    __mat_linear[j, i] = __mat_linear[i, j]
                    process_bar.update(1)
            process_bar.close()
        else:
            for i in range(self.num_pd):
                for j in range(i + 1):
                    __mat_linear[i, j] = np.inner(mat_pi_vec[i], mat_pi_vec[j])
                    __mat_linear[j, i] = __mat_linear[i, j]
        return __mat_linear

    def gram(self):
        mat_linear = self.__matrix_linear()
        mat_gram, self.mat_distance, self.val_tau = tda.make_mat_gram(
            mat_linear, self.name_rkhs)
        return mat_gram

    def make_image(self, mat_pd):
        vec_birth = mat_pd[:, 0]
        vec_death = mat_pd[:, 1]
        return ndimage.filters.gaussian_filter(
            np.multiply(
                self.mat_weight,
                np.histogram2d(vec_birth, vec_death, bins=self.bins)[0]),
            sigma=self.val_sigma_scaled)

    def kernel(self, mat_pd_1, mat_pd_2, name_rkhs=None):
        if name_rkhs is None:
            name_rkhs = self.name_rkhs
        else:
            pass

        vec_pi_1 = self.make_image(mat_pd_1).reshape(-1)
        vec_pi_2 = self.make_image(mat_pd_2).reshape(-1)
        val_c = np.inner(vec_pi_1, vec_pi_2)

        if name_rkhs == "Gaussian":
            val_a = np.inner(vec_pi_1, vec_pi_1)
            val_b = np.inner(vec_pi_2, vec_pi_2)
            return np.exp(-(val_a + val_b - 2.0 * val_c) / (2.0 * self.val_tau))
        else:
            return val_c
