import numpy as np
import pandas as pd
import subprocess
import os
from sklearn.model_selection import StratifiedKFold, KFold

MAX_N = 1.5e4
N_NUM = 3
MAX_M = 40
M_NUM = 5
N_CLASSES = 2

bashCommand = "rm -rf data"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
os.mkdir('data')

n_list = np.linspace(50, MAX_N, num=N_NUM, dtype=int)
m_list = np.linspace(2, MAX_M, num=M_NUM, dtype=int)

mean = [0, 0]
cov = [[0.01, 0], [0, 0.01]]  # diagonal covariance

skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=False)
for n in n_list:
    for m in m_list:
        cov = np.diag(np.full(m, 0.01))
        names_list = ['x{}'.format(i) for i in range(0, m)] + ['label']
        df = pd.DataFrame(columns = names_list)
        for calss_id in range(0, N_CLASSES):
            print("n{}m{}id{}".format(n, m, calss_id))
            mean = np.random.uniform(-0.5, 0.5, m)
            points = np.random.multivariate_normal(mean, cov, n)
            labels = np.full((n, 1), calss_id, dtype=int)
            tmpdf = pd.DataFrame(data=np.hstack((points, labels)), columns = names_list) 
            df = df.append(tmpdf, ignore_index = True)
        df.to_csv("data/n{}m{}full.csv".format(n*N_CLASSES, m), index=False)
        for fold_id, (train_index, test_index) in enumerate(skf.split(df['x0'].tolist(), 
                                                                      df['label'].tolist())):
            tmp_df_train = df.iloc[train_index]
            tmp_df_test = df.iloc[test_index]
            tmp_df_train.to_csv("data/n{}m{}train{}.csv".format(n*N_CLASSES, m, fold_id), index=False)
            tmp_df_test.to_csv("data/n{}m{}test{}.csv".format(n*N_CLASSES, m, fold_id), index=False)

        

