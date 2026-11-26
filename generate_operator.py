"""Generate the original dense 60-view Radon matrix (128 x 128)."""
import argparse
from pathlib import Path
import numpy as np
import time
from skimage.transform import rotate

def weight(N, thetas):
    I = N * N                         # 256，256
    J = len(thetas)                 # 60
    A = np.zeros(shape=(N, N))      # 256，256
    W = np.zeros(shape=(N * J, I))    # 256*60，256*256
    Q = np.zeros(shape=(N, N, N))   # 256，256，256
    for th in range(J):             # 每个角度循环
        start_time = time.time()
        theta = thetas[th]
        for i in range(N):          # 第i行
            # print(i)
            for j in range(N):      # 第j列
                A[i, j] = 1
                R = rotate(A, -theta)  # 旋转矩阵A，角度为-theta
                aux = np.sum(R, 1)  # R中每行元素累加
                Q[i, j, :] = np.squeeze(aux)  # Q[I,J]的值为 R中每行元素累加
                A[i, j] = 0
        end_time = time.time()
        print(end_time - start_time)
        l = np.size(Q, 2)  # 256
        for k in range(l):
            s = l * th + k              # 256*theta+k
            W[s, :] = np.reshape(np.squeeze(Q[:, :, k]), [1, N * N])
    return W

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='R_limited.npy')
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,weight(128,np.linspace(0,135,60)))
