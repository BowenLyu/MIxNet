import numpy as np

def points_scale(points):
    # translate to the center and scale to 1
    points = np.asfarray(points)
    points = points - points.mean(axis=0,keepdims=True)
    scale = np.abs(points).max()
    points = points/scale
    return points
