import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Adding parent folder to path (Python3 workaround)
import sys
sys.path.append('..')

from utils.datagen_utils import Gaussian
# sns.distplot(Gaussian(0, 0.25, 10000).generate(), color="blue", hist=False, rug=True);
# sns.distplot(Gaussian(1, 0.25, 10000).generate(), color="red", hist=False, rug=True);




mean, cov, mean2 = [0, 0], [(2, 0),(0, 2)], [4,4]

data = np.random.multivariate_normal(mean, cov, 1000)
data2 = np.random.multivariate_normal(mean2, cov, 1000)
df = pd.DataFrame(data, columns=["x", "y"])

data3 = np.append(data,data2, 0)
df2 = pd.DataFrame(data3, columns=["x", "y"])

x = np.asarray([data[0][0], data[0][1]])
x = np.reshape(x, (1,-1))
covar = np.asarray(cov)

print(x.dot(covar).dot(x.transpose()))


# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.gca().set_aspect('equal', adjustable='box')
sns.jointplot(x="x", y="y",kind="kde", data=df2);

plt.show()
