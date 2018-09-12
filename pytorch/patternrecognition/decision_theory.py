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
sns.distplot(Gaussian(0, 0.25, 10000).generate(), color="blue", hist=False, rug=True);
sns.distplot(Gaussian(1, 0.25, 10000).generate(), color="red", hist=False, rug=True);




mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

sns.jointplot(x="x", y="y",kind="kde", data=df);

plt.show()
