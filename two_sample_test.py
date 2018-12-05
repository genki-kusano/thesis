import argparse
import numpy as np
import os
import random

# original functions
from functions import tda as tda
from functions import kernel as kernel


def n_mmd(mat_gram_total, unbias=True):
    num_pd_total = mat_gram_total.shape[0]
    idx_divide = num_pd_total // 2

    mat_xx = mat_gram_total[0:idx_divide, 0:idx_divide]
    mat_yy = mat_gram_total[idx_divide:num_pd_total, idx_divide:num_pd_total]
    mat_xy = mat_gram_total[0:idx_divide, idx_divide:num_pd_total]
    val_xx = sum(sum(mat_xx))
    val_yy = sum(sum(mat_yy))
    val_xy = sum(sum(mat_xy))
    if unbias:
        val_xx -= sum(np.diag(mat_xx))
        val_yy -= sum(np.diag(mat_yy))
        return (val_xx + val_yy) / (idx_divide - 1) - (2 * val_xy / idx_divide)
    else:
        return (val_xx + val_yy - 2 * val_xy) / idx_divide


def hist_wchi(mat_gram_total, num_hist=int(1e+4)):
    num_pd_total = mat_gram_total.shape[0]

    # centered Gram matrix
    mat_center = np.empty((num_pd_total, num_pd_total))
    vec_gram = sum(mat_gram_total)
    val_total = sum(vec_gram)
    for i in range(num_pd_total):
        for j in range(i + 1):
            mat_center[i, j] = (mat_gram_total[i, j]
                                - ((vec_gram[i] + vec_gram[j]) / num_pd_total)
                                + (val_total / (num_pd_total ** 2)))
            mat_center[j, i] = mat_center[i, j]

    # estimated eigenvalues
    vec_nu = np.sort(np.linalg.eigh(mat_center)[0])[::-1][0: - 1]
    vec_lambda = vec_nu / (num_pd_total - 1)

    # histogram of the null distribution (weighted chi square)
    vec_hist = np.empty(num_hist)
    for i in range(num_hist):
        vec_z = np.power(np.random.normal(0, np.sqrt(2), num_pd_total - 1), 2)
        vec_hist[i] = np.inner(vec_lambda, vec_z) - 2 * sum(vec_lambda)

    return np.sort(vec_hist)[::-1]


def extract_submatrix(mat_gram_total, num_resample=None):
    num_pd_total = mat_gram_total.shape[0]
    idx_divide = num_pd_total // 2

    if num_resample is None:
        num_resample = idx_divide - 10
    else:
        pass

    d = int(2 * num_resample)
    mat = np.empty((d, d))
    idx_x = random.sample(range(0, idx_divide), num_resample)
    idx_y = random.sample(range(idx_divide, num_pd_total), num_resample)
    idx_xy = idx_x + idx_y
    for i, a in enumerate(idx_xy):
        for j, b in enumerate(idx_xy):
            mat[i, j] = mat_gram_total[a, b]
    return mat


def acceptance_ratio(mat_gram_total, val_alpha=0.05, num_resample=None,
                     num_trial=int(1e+3)):
    vec_wchi = hist_wchi(mat_gram_total)
    num_quantile = int(val_alpha * len(vec_wchi) + 1)
    val_quantile = vec_wchi[num_quantile]

    vec_mmd = np.empty(num_trial)
    for idx_test in range(num_trial):
        mat_reduced = extract_submatrix(mat_gram_total, num_resample)
        vec_mmd[idx_test] = n_mmd(mat_reduced)
    vec_accept = np.where(vec_mmd < val_quantile)[0]
    return len(vec_accept) / num_trial


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
    parser.add_argument("--num_pd", default=100, type=int)
    parser.add_argument("--dim_pd", default=1, type=int)
    parser.add_argument("--scale", default=True,
                        help="do you scale squared birth-death coordinates?")
    parser.add_argument("--gram", default="pwk")
    parser.add_argument("--kernel", default="Gaussian")
    parser.add_argument("--weight", default="arctan")
    parser.add_argument("--sigma", default=None, type=float)
    parser.add_argument("--c", default=None, type=float)
    parser.add_argument("--p", default=5, type=float)
    parser.add_argument("--rkhs", default="Gaussian")

    parser.add_argument("--approx", default=True)
    parser.add_argument("--tqdm", default=True, type=bool)

    parser.add_argument("--use", default=80, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)

    return parser.parse_args()


def main():
    args = parser_()
    name_dir_pcd_p = args.path_p
    name_dir_pcd_q = args.path_q
    num_pd = args.num_pd
    dim_pd = args.dim_pd
    scale = args.scale

    name_gram = args.gram
    name_kernel = args.kernel
    name_weight = args.weight
    val_sigma = args.sigma
    val_c = args.c
    val_p = args.p
    name_rkhs = args.rkhs

    approx = args.approx
    tqdm_bar = args.tqdm

    num_pd_use = args.use
    val_alpha = args.alpha

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

    """ 
    computing the Gram matrix of {D_1,...,D_n,E_1,...,E_n} 
    acceptance_ratio(mat_gram, val_alpha) is the acceptance ratio of P = Q, but 
    the truth is P ¥neq Q, i.e., acceptance_ratio(mat_gram, val_alpha) is the 
    type II error under P = Q
    """
    mat_gram, name_mat = kernel.gram(
        list_pd, name_gram, name_rkhs, name_kernel, val_sigma, name_weight,
        val_c, val_p, approx, tqdm_bar=tqdm_bar)
    val_type2_error = acceptance_ratio(mat_gram, val_alpha)

    """ 
    computing the Gram matrix of {D_1,...,D_n,D_1,...,D_n} 
    acceptance_ratio(mat_gram_same, val_alpha) is the acceptance ratio of 
    P ¥neq Q, but the truth is P = Q, i.e., 
    acceptance_ratio(mat_gram_same, val_alpha) is the type I error under P = Q
    """
    list_pd_same = []
    list_pd_same.extend(list_pd_p)
    list_pd_same.extend(list_pd_p)
    mat_gram_same, _ = kernel.gram(
        list_pd_same, name_gram, name_rkhs, name_kernel, val_sigma, name_weight,
        val_c, val_p, approx=approx, tqdm_bar=tqdm_bar)
    val_type1_error = 1 - acceptance_ratio(mat_gram_same, val_alpha)

    print(name_mat)
    print(val_type1_error, val_type2_error)


if __name__ == "__main__":
    main()
