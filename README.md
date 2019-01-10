Here are codes about the kernel method for persistence diagrams (support vector machine, kernel PCA, kernel two sample test, uniform confidence interval)

# General comments

## variables
You can change `name_dir_pcd` to a folder containing your persistence diagrams.

## data
Example persistence diagrams are able to be obtained from https://github.com/genki-kusano/point_cloud_data (called `PCD code`)
To run codes in this directly without any change, please run the PCD code creating point sets and persistence diagrams.

# pca.py
For n persistence diagrams D_1,...,D_n, we compute the kernel principal component analysis projection to 2 or 3 dimensional Euclidean space.
If you do not prepare your persistence diagrams, this will compute for `torus` data of the PCD code.

# two_sample_test.py
For two datasets {D_1,...,D_n} and {E_1,...,E_n} of persistence diagrams, we compare the two datasets by the two sample test and compute the type I and type II errors.
If you do not prepare your persistence diagrams, this will compute for some spatial point process data, which are in the `matern` folder (Poisson point process and Matern hard-core point process) of the PCD code.
You also can use the `lattice` data of the PCD code, which are perturbed lattice by several different noise level.

# confidence_band.py
For two datasets {D_1,...,D_n} and {E_1,...,E_n} of persistence diagrams, we compute the confidence intervals for a functional form of the PWK vector by the bootstrap method and compare the two intervals.
The output matrix represents whether the two intervals have intersections each other or not.
If you do not prepare your persistence diagrams, this will compute for `matern` of the PCD code.
You also can use the `lattice` data of the PCD code, which are perturbed lattice by several different noise level.

# svm.py
For n+m persistence diagrams D_1,...,D_n,D_{n+1},...,D_{n+m} and labels t_1,...,t_{n+m} \in {1,-1}, we train a SVM classifier by the former n persistence diagrams and n labels, estimate the labels of the later m persistence diagrams by the classifier, and compute the classification accuracies.
If you do not prepare your persistence diagrams, this will compute for `circle` of the PCD code.
Then, please also prepare the labels by svm_prepare.py in this case.
