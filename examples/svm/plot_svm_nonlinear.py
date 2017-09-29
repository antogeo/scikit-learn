"""
==============
Non-linear SVM
==============

Perform binary classification using non-linear SVC
with RBF kernel. The target to predict is a XOR of the
inputs.

The color map illustrates the decision function learned by the SVC.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

%matplotlib
# np.random.seed(0)
# X = np.random.randn(300, 2)
# Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
DataTable = np.genfromtxt('/home/coma_meth/Dropbox/Lizette_yorgos/train_allFeat.csv',delimiter=',',dtype=None)[1:]
X, y = (DataTable[:,1:3]).astype(np.float), (DataTable[:,0]=='1')
xx, yy = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 500),
np.linspace(min(X[:, 1]), max(X[:, 1]), 500))

# fit the model
clf = svm.NuSVC()
clf.fit(X, y)

# plot the decision function for each datapoint on the grid
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min()-0.1, xx.max()+0.1, yy.min()-.1, yy.max()+.1), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired,
            edgecolors='k')
#plt.xticks(())
#plt.yticks(())
plt.axis([min(X[:, 0])-.01, max(X[:, 0])+.01, min(X[:, 1])-.01, max(X[:, 1])+.01])
plt.show()
