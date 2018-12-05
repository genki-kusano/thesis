from functions import tda as tda
import numpy as np
import os


# only for circle_svm
def label_z1(__list_pd):
    def __label_check(__mat_pd, val_b=1, val_d=4):
        num_points = __mat_pd.shape[0]
        val_label = -1
        for i in range(num_points):
            if (__mat_pd[i, 0] < val_b) and (__mat_pd[i, 1] > val_d):
                val_label = 1
                break
            else:
                pass
        return val_label

    num_pd = len(__list_pd)
    __vec_label_z1 = np.empty(num_pd)
    for k in range(num_pd):
        __vec_label_z1[k] = __label_check(__list_pd[k])
    return __vec_label_z1


def main():
    name_dir_pcd = "%s/Desktop/data_tda/circle/pcd3_num200" % \
                   os.path.expanduser('~')
    num_pd = 200
    dim_pd = 1
    list_pd = tda.make_list_pd(name_dir_pcd, num_pd, dim_pd)

    # label
    vec_label_z2 = np.loadtxt("%s/label_z2.txt" % name_dir_pcd)
    vec_label_z1 = label_z1(list_pd)
    vec_label = vec_label_z1 * vec_label_z2

    np.savetxt("%s/label_z1.txt" % name_dir_pcd, vec_label_z1, delimiter='\t')
    np.savetxt("%s/label.txt" % name_dir_pcd, vec_label, delimiter='\t')


if __name__ == "__main__":
    main()
