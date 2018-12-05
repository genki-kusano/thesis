from tqdm import tqdm
import numpy as np


def projection_to_diagonal(mat_pd):
    num_point = mat_pd.shape[0]
    mat_pr = np.empty((num_point, 2))
    for i in range(num_point):
        val = sum(mat_pd[i, :]) / 2
        mat_pr[i, :] = np.array(val, val)
    return mat_pr


class Kernel:
    def __init__(self, list_pd, num_sw=36, tqdm_bar=False):
        self.__list_pd = list_pd
        self.num_pd = len(list_pd)
        self.num_sw = num_sw
        self.tqdm_bar = tqdm_bar

        self.mat_distance = None
        self.val_tau = None

    def sw_distance(self, mat_pd_1, mat_pd_2):
        mat_pr_1 = projection_to_diagonal(mat_pd_1)
        mat_pr_2 = projection_to_diagonal(mat_pd_2)

        mat_1 = np.r_[mat_pd_1, mat_pr_2]
        mat_2 = np.r_[mat_pd_2, mat_pr_1]

        val_sw = 0
        for i in range(self.num_sw):
            val_theta = np.pi * ((i / self.num_sw) - (1 / 2))
            vec_circle = np.array([np.cos(val_theta), np.sin(val_theta)])
            vec_inner_1 = np.sort(np.inner(mat_1, vec_circle))
            vec_inner_2 = np.sort(np.inner(mat_2, vec_circle))
            val_sw += np.linalg.norm(vec_inner_1 - vec_inner_2, 1)

        return val_sw / np.pi

    def __matrix_distance(self):
        __mat_distance = np.empty((self.num_pd, self.num_pd))
        if self.tqdm_bar:
            process_bar = tqdm(total=int(self.num_pd * (self.num_pd + 1) / 2))
            for i in range(self.num_pd):
                for j in range(i + 1):
                    process_bar.set_description("sw: (%s, %s)" % (i, j))
                    __mat_distance[i, j] = self.sw_distance(
                        self.__list_pd[i], self.__list_pd[j])
                    __mat_distance[j, i] = __mat_distance[i, j]
                    process_bar.update(1)
            process_bar.close()
        else:
            for i in range(self.num_pd):
                for j in range(i + 1):
                    __mat_distance[i, j] = self.sw_distance(
                        self.__list_pd[i], self.__list_pd[j])
                    __mat_distance[j, i] = __mat_distance[i, j]
        return __mat_distance

    def gram(self):
        self.mat_distance = self.__matrix_distance()
        max_mat = self.mat_distance.max()
        if (max_mat > 1e+3) or (max_mat < 1e-3):
            self.mat_distance /= max_mat
        else:
            pass

        vec_tau = np.empty(int(self.num_pd * (self.num_pd - 1) / 2))
        idx_temp = 0
        for i in range(self.num_pd):
            for j in range(i):
                vec_tau[idx_temp] = self.mat_distance[i, j]
                idx_temp += 1
        self.val_tau = np.median(vec_tau)

        mat_gram = np.empty((self.num_pd, self.num_pd))
        for i in range(self.num_pd):
            for j in range(i + 1):
                mat_gram[i, j] = np.exp(
                    -1.0 * self.mat_distance[i, j] / (2.0 * self.val_tau))
                mat_gram[j, i] = mat_gram[i, j]
        return mat_gram

    def kernel(self, mat_pd_1, mat_pd_2):
        val_distance = self.sw_distance(mat_pd_1, mat_pd_2)
        return np.exp(-1.0 * val_distance / (2.0 * self.val_tau))
