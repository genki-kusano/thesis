from scipy import optimize
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
import os
import random

# original functions
from functions import tda as tda
from functions import kernel as kernel


# list_mat_pwk(list_pd, k, w, x, L, N)[k][i,j]
# = V^{k,w}(list_pd[k])(x_i + i * L /N, x_2 + j * L /N)
def list_mat_pwk(list_pd, func_kernel, func_weight, vec_x, val_el, num_mesh):
    num_pd = len(list_pd)

    # prepare a grid matrix
    mat_grid = np.empty((num_mesh, num_mesh, 2))
    for i in range(num_mesh):
        for j in range(num_mesh):
            mat_grid[i, j, 0] = vec_x[0] + val_el * (2 * i / num_mesh - 1)
            mat_grid[i, j, 1] = vec_x[1] + val_el * (2 * j / num_mesh - 1)

    # prepare weight vectors
    def __vector_weight(_mat_pd):
        num_point = _mat_pd.shape[0]
        _vec_weight = np.empty(num_point)
        for _i in range(num_point):
            _vec_weight[_i] = func_weight(_mat_pd[_i, :])
        return _vec_weight

    list_weight = []
    for k in range(num_pd):
        list_weight.append(__vector_weight(list_pd[k]))

    # compute pwk values
    def __matrix_pwk(_mat_pd, _vec_weight):
        def __value_pwk(z):
            num_points = _mat_pd.shape[0]
            vec_k = np.empty(num_points)
            for __i in range(num_points):
                vec_k[__i] = func_kernel(z, _mat_pd[__i, :])
            return np.inner(_vec_weight, vec_k)

        _mat = np.empty((num_mesh, num_mesh))
        for _i in range(num_mesh):
            for _j in range(num_mesh):
                _mat[_i, _j] = __value_pwk(mat_grid[_i, _j])
        return _mat

    list_mat = []
    process_bar = tqdm(total=num_pd)
    for k in range(num_pd):
        process_bar.set_description("Confidence: %s" % k)
        list_mat.append(__matrix_pwk(list_pd[k], list_weight[k]))
        process_bar.update(1)
    process_bar.close()
    return list_mat


def make_band(list_pd, func_kernel, func_weight, vec_x, val_el, val_alpha=0.05,
              num_mesh=10, num_boot=int(1e+3)):
    num_pd = len(list_pd)

    def __average_list(_list_mat):
        _mat = np.zeros((num_mesh, num_mesh))
        for k in range(num_pd):
            _mat += _list_mat[k]
        return _mat / num_pd

    def __list_resample(_list_mat):
        idx_resample = random.choices(range(num_pd), k=num_pd)
        _list_temp = []
        for k in range(num_pd):
            _list_temp.append(_list_mat[idx_resample[k]])
        return _list_temp

    list_mat = list_mat_pwk(list_pd, func_kernel, func_weight, vec_x, val_el,
                            num_mesh)

    mat_average = __average_list(list_mat)
    vec_boot = np.empty(num_boot)
    for b in range(num_boot):
        mat_boot = __average_list(__list_resample(list_mat))
        vec_boot[b] = np.sqrt(num_pd) * np.max(np.abs(mat_boot - mat_average))

    def func_emp(c):
        cumulative = len(np.where(vec_boot < c)[0]) / len(vec_boot)
        return cumulative - (1 - val_alpha)

    c_hat = optimize.brentq(func_emp, min(vec_boot), max(vec_boot))
    return np.float64(c_hat) / np.sqrt(num_pd), mat_average


def parser_():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_p",
                        default="%s/Desktop/data_tda/matern/"
                                "pcd2_intensity200_radius003_num100/type0" %
                                os.path.expanduser('~'))
    parser.add_argument("--path_q",
                        default="%s/Desktop/data_tda/matern/"
                                "pcd2_intensity200_radius003_num100/type2" %
                                os.path.expanduser('~'))
    parser.add_argument("--path_save",
                        default="%s/Desktop/data_tda/matern/"
                                "pcd2_intensity200_radius003_num100/"
                                "type_0_vs_type2" %
                                os.path.expanduser('~'))
    parser.add_argument("--num_pd", default=100, type=int)
    parser.add_argument("--dim_pd", default=1, type=int)
    parser.add_argument("--scale", default=True,
                        help="do you scale squared birth-death coordinates?")
    parser.add_argument("--kernel", default="Gaussian")
    parser.add_argument("--weight", default="arctan")
    parser.add_argument("--sigma", default=None, type=float)
    parser.add_argument("--c", default=None, type=float)
    parser.add_argument("--p", default=5, type=float)

    parser.add_argument("--use", default=80, type=int)
    parser.add_argument("--x1", default=0.03, type=float)
    parser.add_argument("--x2", default=0.05, type=float)
    parser.add_argument("--el", default=0.02, type=float)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--idx_i", default=-1, type=int)
    parser.add_argument("--idx_j", default=-1, type=int)
    return parser.parse_args()


def main():
    args = parser_()
    name_dir_pcd_p = args.path_p
    name_dir_pcd_q = args.path_q
    name_dir_save = args.path_save
    num_pd = args.num_pd
    dim_pd = args.dim_pd
    scale = args.scale

    name_kernel = args.kernel
    name_weight = args.weight
    val_sigma = args.sigma
    val_c = args.c
    val_p = args.p

    num_pd_use = args.use
    vec_x = np.array([args.x1, args.x2])
    val_el = args.el
    val_alpha = args.alpha

    num_mesh = 20
    idx_i = args.idx_i
    idx_j = args.idx_j

    """ 
    importing persistence diagrams as list 
    list_pd_p = {D_1,...,D_n}
    list_pd_q = {E_1,...,E_n}
    list_pd = {D_1,...,D_n,E_1,...,E_n}
    """
    list_pd_p = tda.make_list_pd(
        name_dir_pcd_p, num_pd, dim_pd, scale=scale)[0:num_pd_use]
    list_pd_q = tda.make_list_pd(
        name_dir_pcd_q, num_pd, dim_pd, scale=scale)[0:num_pd_use]
    list_pd = []
    list_pd.extend(list_pd_p)
    list_pd.extend(list_pd_q)

    """ define several parameters """
    if val_c is None:
        val_c = tda.parameter_birth_death_pers(list_pd)[2]
    else:
        pass
    if val_sigma is None:
        val_sigma_p = tda.parameter_sigma(list_pd_p, name_dir_pcd_p)
        val_sigma_q = tda.parameter_sigma(list_pd_q, name_dir_pcd_q)
        val_sigma = (val_sigma_p + val_sigma_q) / 2
    else:
        pass

    """ define a positive definite kernel and a weight function """
    func_kernel = kernel.function_kernel(name_kernel, val_sigma)
    func_weight = kernel.function_weight(name_weight, val_c, val_p)

    """
    From a persistence diagram D, we make a PWK function 
    f^{k,w}_{z}(D)=V^{k,w}(D)(z) ¥in ¥mathbb{R} 
    (z ¥in [x[0] - el, x[0] + el] ¥times [x[1] - el, x[1] + el])
    mat_average_p[i, j] = (1/n) ¥sum_{k=1}^{n}f^{k,w}_{z_{i, j}}(D_{k})
    where z_{i, j} = (x[0] + el * (2 * i / num_mesh - 1), 
                      x[1] + el * (2 * i / num_mesh - 1))
    val_band_p is the uniform confidence band of {f^{k,w}_{z}(D_i)}
    
    [mat_average_p[i, j] - val_band_p, mat_average_p[i, j] + val_band_p] is
    a confidence interval for Pf^{k,w}_{z_{i,j}}(D) at level 1 - val_alpha
    """
    val_band_p, mat_average_p = make_band(list_pd_p, func_kernel, func_weight,
                                          vec_x, val_el, val_alpha, num_mesh)
    val_band_q, mat_average_q = make_band(list_pd_q, func_kernel, func_weight,
                                          vec_x, val_el, val_alpha, num_mesh)

    """
    If [mat_average_p[i, j] - val_band_p, mat_average_p[i, j] + val_band_p]
    and [mat_average_q[i, j] - val_band_q, mat_average_q[i, j] + val_band_q]
    have no intersection, 
    |mat_average_p[i, j] - mat_average_q[i, j]| - val_band_p + val_band_q
    is positive.
    
    """
    mat_diff = np.abs(mat_average_p - mat_average_q)
    val_band = val_band_p + val_band_q
    mat_check = ((np.sign(mat_diff - val_band) + 1) // 2)[::-1]
    num_error_rate = 1 - sum(sum(mat_check)) / (num_mesh ** 2)

    """
    saving the mat_check
    If mat_check[i, j] is white, there in no intersections between two 
    confidence intervals
    """
    tda.mkdir_os(name_dir_save)
    name_param = "%s_%s" % (name_kernel, name_weight)
    plt.figure()
    plt.imshow(
        mat_check * 255, interpolation='nearest', cmap="gray", vmin=0, vmax=255)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.savefig("%s/mat_%s_error%04d.png" % (
        name_dir_save, name_param, num_error_rate * 1000))
    plt.close()

    """ compare the intervals at vec_x """
    if idx_i == -1:
        idx_i = num_mesh // 2
    else:
        pass
    if idx_j == -1:
        idx_j = num_mesh // 2
    else:
        pass

    vec_ave = [mat_average_p[idx_i, idx_j], mat_average_q[idx_i, idx_j]]
    vec_band = [val_band_p, val_band_q]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.array([0, 1]), vec_ave, yerr=[vec_band, vec_band],
                fmt='ro', ecolor="g", capsize=10, ms=10)
    ax.set_xlim(-1, 2)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("%s/interval_%s.png" % (name_dir_save, name_param))
    plt.close()


if __name__ == "__main__":
    main()
