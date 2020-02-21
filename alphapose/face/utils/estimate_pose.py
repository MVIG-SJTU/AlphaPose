# coding: UTF-8

from math import asin, atan2, cos, sin

import numpy as np
import os

def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 or R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return x, y, z


def P2sRt(P):
    ''' decompositing camera matrix P. 
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation. 
    '''
    t2d = P[:2, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t2d


def compute_similarity_transform(points_static, points_to_transform):
    # http://nghiaho.com/?page_id=671
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3, 1)
    t1 = -np.mean(p1, axis=1).reshape(3, 1)
    t_final = t1 - t0

    p0c = p0 + t0
    p1c = p1 + t1

    covariance_matrix = p0c.dot(p1c.T)
    U, S, V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0) ** 2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0) ** 2))

    s = (rms_d0 / rms_d1)
    P = np.c_[s * np.eye(3).dot(R), t_final]
    return P



current_path = os.path.dirname(__file__)
def estimate_pose(vertices):
    canonical_vertices = np.load(current_path+'/uv_data/canonical_vertices.npy')
    P = compute_similarity_transform(vertices, canonical_vertices)
    _, R, _ = P2sRt(P)  # decompose affine matrix to s, R, t
    pose = matrix2angle(R)

    return P, pose
