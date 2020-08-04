import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X = dataset.data
Y = dataset.target

# print(X.shape)
# print(Y.shape)

# pca = PCA(n_components= 5)
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr)          #0.40242142 0.14923182 0.12059623 0.09554764 0.06621856
# print(sum(pca_evr))     #0.8340156689459766

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_) #cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
print(cumsum)

aaa = np.argmax(cumsum > 0.94) + 1
print(cumsum>=0.94)
print(aaa)