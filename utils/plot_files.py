import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def scatter_plot(path, columns = 2):
    data = np.loadtxt(path, delimiter=",")
    dims = data.shape[1]
    fig = plt.figure(figsize = None)
    for i in range(1,dims):
        plt.subplot(dims//columns + 1,columns , i)
        plt.scatter(data[:,0],data[:,i])
        plt.xlim(-10,10)
        plt.ylim(-10,10)
    plt.show()

def scatter_pca_plot(path, plot_dim = 3):
    data = np.loadtxt(path, delimiter=",")
    dims = data.shape[1]
    X_reduced = PCA(n_components=plot_dim).fit_transform(data)
    fig = plt.figure()
    if plot_dim == 2:
        plt.scatter(X_reduced[:,0],X_reduced[:,1])
    elif plot_dim == 3:
        ax = Axes3D(fig)
        ax.plot(X_reduced[:,0], X_reduced[:, 1], X_reduced[:, 2],marker="x", linestyle='None')

    plt.show()

if __name__ == "__main__":
    scatter_pca_plot("sample.csv")