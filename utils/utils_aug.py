import random
import numpy as np
from random import gauss
from transformations import rotation_matrix
from scipy.ndimage import map_coordinates

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]


def coordinateTransformWrapper(X_T1, maxDeg=0, maxShift=7.5, mirror_prob=0.):
    randomAngle = np.radians(maxDeg * 2 * (random.random() - 0.5))
    unitVec = tuple(make_rand_vector(3))
    shiftVec = [maxShift * 2 * (random.random() - 0.5),
                maxShift * 2 * (random.random() - 0.5),
                maxShift * 2 * (random.random() - 0.5)]
    X_T1 = coordinateTransform(X_T1, randomAngle, unitVec, shiftVec)
    return X_T1


def coordinateTransform(vol, randomAngle, unitVec, shiftVec, order=1, mode='constant'):
    '''
    Implemented based on  https://github.com/benniatli/BrainAgePredictionResNet
    '''
    ax = (list(vol.shape))
    ax = [ax[i] for i in [1, 0, 2]]
    coords = np.meshgrid(np.arange(ax[0]), np.arange(ax[1]), np.arange(ax[2]))

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1) - float(ax[0]) / 2,  # x coordinate, centered
                     coords[1].reshape(-1) - float(ax[1]) / 2,  # y coordinate, centered
                     coords[2].reshape(-1) - float(ax[2]) / 2,  # z coordinate, centered
                     np.ones((ax[0], ax[1], ax[2])).reshape(-1)])  # 1 for homogeneous coordinates

    mat = rotation_matrix(randomAngle, unitVec)

    transformed_xyz = np.dot(mat, xyz)

    x = transformed_xyz[0, :] + float(ax[0]) / 2 + shiftVec[0]
    y = transformed_xyz[1, :] + float(ax[1]) / 2 + shiftVec[1]
    z = transformed_xyz[2, :] + float(ax[2]) / 2 + shiftVec[2]
    x = x.reshape((ax[1], ax[0], ax[2]))
    y = y.reshape((ax[1], ax[0], ax[2]))
    z = z.reshape((ax[1], ax[0], ax[2]))
    new_xyz = [y, x, z]
    new_vol = map_coordinates(vol, new_xyz, order=order, mode=mode)
    return new_vol

if __name__ == '__main__':
    img = np.random.randn(100,100,100)
    img = coordinateTransformWrapper(img, maxDeg=10, maxShift=5, mirror_prob=0)

