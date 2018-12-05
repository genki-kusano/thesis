from sklearn.svm import SVC
from functions import tda as tda
from functions import kernel as kernel
import argparse
import numpy as np
import os


def slice_list(list_pd, range_idx):
    list_pd_sub = []
    for i in range_idx:
        list_pd_sub.append(list_pd[i])
    return list_pd_sub


def divide_gram(mat_gram, range_train):
    range_test = list(set(range(mat_gram.shape[0])) - set(range_train))
    mat_gram_train = mat_gram[range_train, :][:, range_train]
    mat_gram_test = mat_gram[range_test, :][:, range_train]
    return [mat_gram_train, mat_gram_test]


# list_gram = [mat_gram_train, mat_gram_test]
# list_label = [vec_label_train, vec_label_test]
def predict(list_gram, list_label):
    svc = SVC(kernel='precomputed')
    svc.fit(list_gram[0], list_label[0])
    vec_label_pred = svc.predict(list_gram[1])
    vec_error = np.nonzero(vec_label_pred - list_label[1])[0]
    val_error = len(vec_error) / len(list_label[1])
    return 1.0 - val_error, vec_label_pred


def parser_():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        default="%s/Desktop/data_tda/circle/"
                                "pcd3_num200" % os.path.expanduser('~'))
    parser.add_argument("--label",
                        default="%s/Desktop/data_tda/circle/"
                                "pcd3_num200/label.txt" %
                                os.path.expanduser('~'))
    parser.add_argument("--num_pd", default=200, type=int)
    parser.add_argument("--dim_pd", default=1, type=int)
    parser.add_argument("--scale", default=True,
                        help="do you scale squared birth-death coordinates?")
    parser.add_argument("--gram", default="pwk")
    parser.add_argument("--kernel", default="Gaussian")
    parser.add_argument("--weight", default="arctan")
    parser.add_argument("--sigma", default=1, type=float)
    parser.add_argument("--c", default=0.1, type=float)
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

    """ importing labels for persistence diagrams as vector """
    vec_label = np.loadtxt(args.label)

    """ computing the Gram matrix of {D_1,...,D_{n+m}} """
    mat_gram, name_mat = kernel.gram(
        list_pd, name_gram, name_rkhs, name_kernel, val_sigma, name_weight,
        val_c, val_p, approx=approx, num_mesh=num_mesh, tqdm_bar=tqdm_bar)

    """ 
    divide list_pd into {D_1,...,D_m} (training data) and 
    {D_{m+1},...,D_{n+m}} (test data), make the SVM by the training data, and
    calculate the accuracy for the test data
    """
    range_train = range(int(num_pd // 2))
    range_test = list(set(range(len(vec_label))) - set(range_train))
    list_gram = divide_gram(mat_gram, range_train)
    list_label = [vec_label[range_train], vec_label[range_test]]
    val_accuracy = predict(list_gram, list_label)[0]
    print(name_mat)
    print(val_accuracy)


if __name__ == "__main__":
    main()
