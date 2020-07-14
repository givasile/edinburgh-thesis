"""Checking if a point is inside a n-dimensional Bounding Box."""


import scipy.ndimage as ndimage
import scipy.spatial.transform as tr
import numpy as np


class ndimBoundingBox:
    def __init__(self, rotation, center, limits):
        assert rotation.ndim == 2
        assert center.ndim == 1
        
        self.rotation = rotation
        self.center = center
        self.limits = limits
        self.dim = rotation.shape[0]

        self.rotation_inv = np.linalg.inv(self.rotation)

        assert self.dim == center.shape[0]
        assert rotation.shape[0] == self.dim
        assert limits.shape[0] == self.dim

    def contains(self, point):
        assert point.ndim == 1
        assert point.shape[0] == self.dim

        # transform to bb coordinate system
        point1 = np.dot(self.rotation_inv, point) + np.dot(self.rotation_inv, -self.center)
        print(point)
        print(point1)

        
theta = np.radians(45)
rot = np.array(((np.cos(theta), -np.sin(theta)),
                (np.sin(theta),  np.cos(theta))))
center = np.array([1, 1])
limits = np.array([[1, -1], [2, -1]])
point = np.array([0, 2])

bb = ndimBoundingBox(rot, center, limits)
print(bb.contains(point))
