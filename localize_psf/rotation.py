"""
Tools for constructing and working with 3D rotation matrices
"""

from collections.abc import Sequence
import numpy as np


def get_rot_mat(rot_axis: Sequence[float],
                gamma: float) -> np.ndarray:
    """
    Get matrix which rotates points about the specified axis by the given angle. Think of this rotation matrix
    as acting on unit vectors, and hence its inverse R^{-1} transforms regular vectors. Therefore, we define
    this matrix such that it rotates unit vectors in a lefthanded sense about the given axis for positive gamma.
    e.g. when rotating about the z-axis this becomes
    [[cos(gamma), -sin(gamma), 0],
     [sin(gamma), cos(gamma), 0],
     [0, 0, 1]]
    since vectors are acted on by the inverse matrix, they rotated in a righthanded sense about the given axis.

    :param rot_axis: unit vector specifying axis to rotate about, [nx, ny, nz]
    :param float gamma: rotation angle in radians to transform point. A positive angle corresponds right-handed rotation
    about the given axis
    :return mat: 3x3 rotation matrix
    """
    if np.abs(np.linalg.norm(rot_axis) - 1) > 1e-12:
        raise ValueError("rot_axis must be a unit vector")

    nx, ny, nz = rot_axis
    mat = np.array([[nx**2 * (1 - np.cos(gamma)) + np.cos(gamma),
                     nx * ny * (1 - np.cos(gamma)) - nz * np.sin(gamma),
                     nx * nz * (1 - np.cos(gamma)) + ny * np.sin(gamma)],
                    [nx * ny * (1 - np.cos(gamma)) + nz * np.sin(gamma),
                     ny**2 * (1 - np.cos(gamma)) + np.cos(gamma),
                     ny * nz * (1 - np.cos(gamma)) - nx * np.sin(gamma)],
                    [nx * nz * (1 - np.cos(gamma)) - ny * np.sin(gamma),
                     ny * nz * (1 - np.cos(gamma)) + nx * np.sin(gamma),
                     nz**2 * (1 - np.cos(gamma)) + np.cos(gamma)]])
    return mat


def get_rot_mat_angle_axis(rot_mat: np.ndarray) -> (np.ndarray, float):
    """
    Given a rotation matrix, determine the axis it rotates about and the angle it rotates through. This is
    the inverse function for get_rot_mat()

    Note that get_rot_mat_angle_axis(get_rot_mat(axis, angle)) can return either axis, angle or -axis, -angle
    as these two rotation matrices are equivalent

    :param rot_mat:
    :return rot_axis, angle:
    """
    if np.linalg.norm(rot_mat.dot(rot_mat.transpose()) - np.identity(rot_mat.shape[0])) > 1e-12:
        raise ValueError("rot_mat was not a valid rotation matrix")

    eig_vals, eig_vects = np.linalg.eig(rot_mat)

    # rotation matrix must have one eigenvalue that is 1 to numerical precision
    ind = np.argmin(np.abs(eig_vals - 1))

    # construct basis with e3 = rotation axis
    e3 = eig_vects[:, ind].real

    if np.linalg.norm(np.cross(np.array([0, 1, 0]), e3)) != 0:
        e1 = np.cross(np.array([0, 1, 0]), e3)
    else:
        e1 = np.cross(np.array([1, 0, 0]), e3)
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(e3, e1)

    # basis change matrix to look like rotation about z-axis
    mat_basis_change = np.vstack((e1, e2, e3)).transpose()

    # transformed rotation matrix
    r_bc = np.linalg.inv(mat_basis_change).dot(rot_mat.dot(mat_basis_change))
    angle = np.arctan2(r_bc[1, 0].real, r_bc[0, 0].real)

    # the pairs (e3, angle) and (-e3, -angle) represent the same matrix
    # choose the pair so that the largest component of e3 is positive
    ind_max = np.argmax(np.abs(e3))
    if e3[ind_max] < 0:
        e3 = -e3
        angle = -angle

    return e3, angle


def euler_mat(phi: float,
              theta: float,
              psi: float) -> np.ndarray:
    """
    Define our Euler angles connecting the body frame to the space/lab frame by
    r_lab = U_z(phi) * U_y(theta) * U_z(psi) * r_body
    The coordinates are column vectors r = [[x], [y], [z]], so
    U_z(phi) = [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]
    U_y(theta) = [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]

    Consider the z-axis in the body frame. This axis is then orientated at
    [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]
    in the space frame. i.e. phi, theta are the usual polar angles. psi represents a rotation of the object
    about its own axis.

    :param phi:
    :param theta:
    :param psi:
    :return euler_mat: U_z(phi) * U_y(theta) * U_z(psi)
    """
    euler_mat = np.array([[np.cos(phi) * np.cos(theta) * np.cos(psi) - np.sin(phi) * np.sin(psi),
                          -np.cos(phi) * np.cos(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
                           np.cos(phi) * np.sin(theta)],
                          [np.sin(phi) * np.cos(theta) * np.cos(psi) + np.cos(phi) * np.sin(psi),
                          -np.sin(phi) * np.cos(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                           np.sin(phi) * np.sin(theta)],
                          [-np.sin(theta) * np.cos(psi), np.sin(theta) * np.sin(psi), np.cos(theta)]])

    return euler_mat


def euler_mat_inv(phi: float,
                  theta: float,
                  psi: float) -> np.ndarray:
    """
    r_body = U_z(-psi) * U_y(-theta) * U_z(-phi) * r_lab

    :param phi:
    :param theta:
    :param psi:
    :return dphi, dtheta, dsi:
    """
    return euler_mat(-psi, -theta, -phi)


def euler_mat_derivatives(phi: float,
                          theta: float,
                          psi: float) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Derivative of Euler matrix with respect to Euler angles

    :param phi:
    :param theta:
    :param psi:
    :return dphi, dtheta, dsi:
    """
    dphi = np.array([[-np.sin(phi) * np.cos(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                       np.sin(phi) * np.cos(theta) * np.sin(psi) - np.cos(phi) * np.cos(psi),
                      -np.sin(phi) * np.sin(theta)],
                     [ np.cos(phi) * np.cos(theta) * np.cos(psi) - np.sin(phi) * np.sin(psi),
                      -np.cos(phi) * np.cos(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
                       np.cos(phi) * np.sin(theta)],
                     [0, 0, 0]])
    dtheta = np.array([[-np.cos(phi) * np.sin(theta) * np.cos(psi),
                         np.cos(phi) * np.sin(theta) * np.sin(psi),
                         np.cos(phi) * np.cos(theta)],
                       [-np.sin(phi) * np.sin(theta) * np.cos(psi),
                         np.sin(phi) * np.sin(theta) * np.sin(psi),
                         np.sin(phi) * np.cos(theta)],
                       [-np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)]])
    dpsi = np.array([[-np.cos(phi) * np.cos(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
                      -np.cos(phi) * np.cos(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi),
                      0],
                     [-np.sin(phi) * np.cos(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                      -np.sin(phi) * np.cos(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                      0],
                     [np.sin(theta) * np.sin(psi), np.sin(theta) * np.cos(psi), 0]])

    return dphi, dtheta, dpsi


def euler_mat_inv_derivatives(phi: float,
                              theta: float,
                              psi: float) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Derivative of inverse Euler matrix with respect to Euler angles

    :param phi:
    :param theta:
    :param psi:
    :return dphi, dtheta, dpsi:
    """
    d1, d2, d3 = euler_mat_derivatives(-psi, -theta, -phi)
    dphi = -d3
    dtheta = -d2
    dpsi = -d1

    return dphi, dtheta, dpsi
