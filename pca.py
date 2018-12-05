from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np
import os

# original functions
from functions import tda
from functions import kernel


def kpca(mat_gram):
    num_pd = mat_gram.shape[0]
    mat_center = np.empty((num_pd, num_pd))
    vec_gram = np.sum(mat_gram, 0)
    val_gram = np.sum(vec_gram)

    for i in range(num_pd):
        for j in range(i + 1):
            mat_center[i, j] = (mat_gram[i, j]
                                - ((vec_gram[i] + vec_gram[j]) / num_pd)
                                + (val_gram / (num_pd ** 2)))
            mat_center[j, i] = mat_center[i, j]

    vec_eigen, mat_eigen = np.linalg.eigh(mat_center)
    idx_eigen = vec_eigen.argsort()[::-1]
    vec_eigen = vec_eigen[idx_eigen]
    mat_eigen = mat_eigen[:, idx_eigen]

    mat_kpca = np.empty((3, num_pd))
    for k in range(num_pd):
        mat_kpca[0, k] = np.sqrt(vec_eigen[0]) * mat_eigen[k, 0]
        mat_kpca[1, k] = np.sqrt(vec_eigen[1]) * mat_eigen[k, 1]
        mat_kpca[2, k] = np.sqrt(vec_eigen[2]) * mat_eigen[k, 2]
    return mat_kpca, vec_eigen


def kfdr(mat_gram, val_gamma=1e-4):
    num_pd = mat_gram.shape[0]
    vec_kfdr = np.empty(num_pd - 1)

    for k in range(1, num_pd):
        vec_one1 = np.r_[np.ones(k), np.zeros(num_pd - k)]
        vec_one2 = np.r_[np.zeros(k), np.ones(num_pd - k)]
        vec_eta = - vec_one1 / k + vec_one2 / (num_pd - k)
        mat_q = ((np.matrix(vec_one1).T * vec_one1) / k
                 + (np.matrix(vec_one2).T * vec_one2) / (num_pd - k))
        mat_one = np.identity(num_pd)
        mat_inv = np.linalg.inv(
            mat_gram * (mat_one - mat_q) + val_gamma * mat_one)
        vec_kfdr[k - 1] = (k * (num_pd - k) / num_pd) * (
            vec_eta * mat_inv * mat_gram * np.matrix(vec_eta).T)
    return vec_kfdr


def plot(mat_gram, name_dir_pca=None, name_mat=None):
    num_pd = mat_gram.shape[0]
    vec_kfdr = kfdr(mat_gram)
    idx_kfdr = vec_kfdr.argmax() + 1

    mat_kpca, vec_eigen = kpca(mat_gram)
    mat_before = mat_kpca[:, 0:idx_kfdr]
    mat_after = mat_kpca[:, idx_kfdr:num_pd]
    len_after = num_pd - idx_kfdr - 1

    plt.figure(figsize=(6, 6))
    plt.rcParams["font.size"] = 12
    scale = 0.7
    plt.annotate("PD(1)",
                 xy=(mat_before[0, 0], mat_before[1, 0]),
                 xytext=(scale * mat_before[0, 0], scale * mat_before[1, 0]),
                 arrowprops=dict(facecolor="black", width=0.1, headwidth=0))
    plt.annotate("PD(%s)" % num_pd,
                 xy=(mat_after[0, len_after], mat_after[1, len_after]),
                 xytext=(scale * mat_after[0, len_after],
                         scale * mat_after[1, len_after]),
                 arrowprops=dict(facecolor="black", width=0.1, headwidth=0))

    plt.plot(mat_before[0, :], mat_before[1, :], "bx",
             mat_after[0, :], mat_after[1, :], "ro")
    plt.tick_params(labelbottom="off", bottom="off")
    plt.tick_params(labelleft="off", left="off")
    plt.title("contribution rate: %.2f" %
              float(100 * vec_eigen[0:2].sum() / vec_eigen.sum()))
    plt.savefig("%s/2d_%s_kfdr%s.png" % (name_dir_pca, name_mat, idx_kfdr + 1))
    plt.close()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(mat_before[0, :], mat_before[1, :], mat_before[2, :], "bx")
    ax.plot(mat_after[0, :], mat_after[1, :], mat_after[2, :], "ro")
    ax.text(scale * mat_before[0, 0], scale * mat_before[1, 0],
            scale * mat_before[2, 0], "PD(1)")
    ax.text(scale * mat_after[0, len_after], scale * mat_after[1, len_after],
            scale * mat_after[2, len_after], "PD(%s)" % num_pd)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.title("contribution rate: %.2f" %
              float(100 * vec_eigen[0:3].sum() / vec_eigen.sum()))
    plt.savefig("%s/3d_%s_kfdr%s.png" % (name_dir_pca, name_mat, idx_kfdr + 1))
    plt.close()


def parser_():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        default="%s/Desktop/data_tda/torus/pcd3_sample1000"
                                "_num40" % os.path.expanduser('~'))
    parser.add_argument("--num_pd", default=40, type=int)
    parser.add_argument("--dim_pd", default=1, type=int)
    parser.add_argument("--scale", default=True)
    parser.add_argument("--gram", default="pwk")
    parser.add_argument("--kernel", default="Gaussian")
    parser.add_argument("--weight", default="arctan")
    parser.add_argument("--sigma", default=None, type=float)
    parser.add_argument("--c", default=None, type=float)
    parser.add_argument("--p", default=5, type=float)
    parser.add_argument("--rkhs", default="Gaussian")

    parser.add_argument("--approx", default=True)
    parser.add_argument("--mesh", default=80, type=int)
    parser.add_argument("--tqdm", default=True)
    return parser.parse_args()


def main():
    args = parser_()
    name_dir_pcd = args.path
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
    num_mesh = args.mesh
    tqdm_bar = args.tqdm

    """ importing persistence diagrams as list """
    list_pd = tda.make_list_pd(name_dir_pcd, num_pd, dim_pd, scale=scale)

    """ define several parameters """
    name_c = ""
    if val_c is None:
        val_c = tda.parameter_birth_death_pers(list_pd)[2]  # median pers
    elif val_c == 0:
        val_c = tda.parameter_birth_death_pers(list_pd)[3] / 2  # max pers
        name_c = "_max_pers"
    else:
        pass
    if val_sigma is None:
        val_sigma = tda.parameter_sigma(list_pd, name_dir_pcd)
    else:
        pass

    """ computing the Gram matrix """
    mat_gram, name_mat = kernel.gram(
        list_pd, name_gram, name_rkhs, name_kernel, val_sigma, name_weight,
        val_c, val_p, approx=approx, num_mesh=num_mesh, tqdm_bar=tqdm_bar)

    """ computing the kernel PCA """
    name_mat = "%s%s" % (name_mat, name_c)
    name_dir_pca = "%s/pca" % name_dir_pcd
    tda.mkdir_os(name_dir_pca)
    plot(mat_gram, name_dir_pca, name_mat)


if __name__ == "__main__":
    main()
