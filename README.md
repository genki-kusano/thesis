Here are codes about the kernel method for persistence diagrams (support vector machine, kernel PCA, kernel two sample test, uniform confidence interval) and also a suppelemtary material of my doctoral thesis, which will be uploaded in March 2019.

# General comments

## data
Example persistence diagrams are able to be obtained from https://github.com/genki-kusano/point_cloud_data (called `PCD code`)
To run the codes in this directly without any change, please run the PCD code creating point sets and persistence diagrams.

## variables
You can change `name_dir_pcd` to a folder containing your persistence diagrams.
To run the codes by default setting, please prepare the PCD folder on your desktop. 

# functions folder
The `functions folder` contains 5 methods for persistence diagrams.
* pwk.py  
class Kernel : persistence weighted (Gaussian) kernel (http://jmlr.org/papers/v18/17-317.html)    
class KernelPss : persistence scale-scape kernel (https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Reininghaus_A_Stable_Multi-Scale_2015_CVPR_paper.html)  
As mentioned in , the persistence scale-space kernel can be viewed as an example of the persistence weighted kernel. Hence, we could apply the same approximation method (`self.approx`) by the random Fourier features (see def make_rff).
* landscape.py  
persistence landscape (http://www.jmlr.org/papers/v16/bubenik15a.html)  
* image.py  
persistence image (http://jmlr.org/papers/v18/16-337.html)   
* sw.py  
Sliced Wasserstein kernel (http://proceedings.mlr.press/v70/carriere17a.html)   

* kernel.py  
All above methods can create their Gram matrix by their kernels.


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
