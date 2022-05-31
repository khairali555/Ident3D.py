import numpy as np
# from sklearn.decomposition import PCA
from wpca import WPCA
from math import sin, cos, pi

R3_EPSILON = 1e-8

def R3Normal(v):
    """Return an arbitrary normal to the vector v"""
    if (
        abs(v[0]) <= R3_EPSILON and
        abs(v[1]) <= R3_EPSILON and
        abs(v[2]) <= R3_EPSILON
    ):
        return np.array([1., 0., 0.])
    if (abs(v[0]) >= abs(v[1])):
        if (abs(v[0]) >= abs(v[2])):
            return np.array([-v[1], v[0], 0.])  # x maximal
        else:
            return np.array([v[2], 0., -v[0]])  # z maximal
    else:
        if (fabs(v[1]) >= fabs(v[2])):
            return np.array([v[1], -v[0], 0.])  # y maximal
        else:
            return np.array([0., v[2], -v[1]])  # z maximal

def normalize(v):
    l = np.linalg.norm(v)
    if l > 0.:
        v[0] /= l
        v[1] /= l
        v[2] /= l
    return v

def normalized(v):
    res = v.copy()
    res.normalize()
    return res

def rotationMatrix(axis, angle):
    m = np.array([
        [0.]*3, [0.]*3, [0.]*3
    ])
    if np.linalg.norm(axis) <= R3_EPSILON:
        m[0, 0] = 1.
        m[1, 1] = 1.
        m[2, 2] = 1.
        return m;

    e3 = np.array([axis[0], axis[1], axis[2]], dtype="float64")
    normalize(e3)
    e1 = R3Normal(axis); normalize(e1)
    e2 = np.cross(e3, e1)
    # print("e1 =", e1)
    # print("e2 =", e2)
    # print("e3 =", e3)

    # In basis (e1, e2, e3) the rotation matrix is
    # ( cos(angle)    -sin(angle)    0 )
    # ( sin(angle)    cos(angle)     0 )
    # ( 0             0              1 )

    cosAlpha = cos(angle)
    sinAlpha = sin(angle)

    ex = np.array([1., 0., 0.])
    ey = np.array([0., 1., 0.])
    ez = np.array([0., 0., 1.])

    ex_x = ex @ e1
    ex_y = ex @ e2
    ex_z = ex @ e3

    # Rotate ex
    exRot_x = cosAlpha * ex_x - sinAlpha * ex_y
    exRot_y = sinAlpha * ex_x + cosAlpha * ex_y
    exRot_z = ex_z

    exRot = e1 * exRot_x + e2 * exRot_y + e3 * exRot_z
    # print("exRot =", exRot)

    ey_x = ey @ e1
    ey_y = ey @ e2
    ey_z = ey @ e3

    # Rotate ey
    eyRot_x = cosAlpha * ey_x - sinAlpha * ey_y
    eyRot_y = sinAlpha * ey_x + cosAlpha * ey_y
    eyRot_z = ey_z

    eyRot = e1 * eyRot_x + e2 * eyRot_y + e3 * eyRot_z
    # print("eyRot =", eyRot)

    ez_x = ez @ e1
    ez_y = ez @ e2
    ez_z = ez @ e3

    # Rotate ez
    ezRot_x = cosAlpha * ez_x - sinAlpha * ez_y
    ezRot_y = sinAlpha * ez_x + cosAlpha * ez_y
    ezRot_z = ez_z

    ezRot = e1 * ezRot_x + e2 * ezRot_y + e3 * ezRot_z
    # print("ezRot =", ezRot)

    m[0, 0] = exRot[0]; m[0, 1] = eyRot[0]; m[0, 2] = ezRot[0]
    m[1, 0] = exRot[1]; m[1, 1] = eyRot[1]; m[1, 2] = ezRot[1]
    m[2, 0] = exRot[2]; m[2, 1] = eyRot[2]; m[2, 2] = ezRot[2]
    return m

def defineTransform(model1, weight1, model2, weight2):
    pca1 = WPCA()
    pca1.fit(model1, weight1)
    # print("components 1:", pca1.components_)
    # print("variances 1:", pca1.explained_variance_)

    pca2 = WPCA()
    pca2.fit(model2, weight2)
    # print("components 2:", pca2.components_)
    # print("variances 2:", pca2.explained_variance_)

    secondMoment1 = np.array([0.]*3)
    weight2Sum = 0.
    for i in range(len(model1)):
        w2 = weight1[i]**2
        weight2Sum += w2
        v = model1[i] - pca1.mean_
        v *= np.linalg.norm(v)
        secondMoment1 += v*w2
    secondMoment1 /= weight2Sum

    e1x = pca1.components_[0].copy()
    if e1x @ secondMoment1 < 0.:
        e1x = -e1x
    e1y = pca1.components_[1].copy()
    if e1y @ secondMoment1 < 0.:
        e1y = -e1y
    e1z = pca1.components_[2].copy()
    if e1z @ secondMoment1 < 0.:
        e1z = -e1z

    secondMoment2 = np.array([0.]*3)
    weight2Sum = 0.
    for i in range(len(model2)):
        w2 = weight2[i]**2
        weight2Sum += w2
        v = model2[i] - pca2.mean_
        v *= np.linalg.norm(v)
        secondMoment2 += v*w2
    secondMoment2 /= weight2Sum

    e2x = pca2.components_[0].copy()
    if e2x @ secondMoment2 < 0.:
        e2x = -e2x
    e2y = pca2.components_[1].copy()
    if e2y @ secondMoment2 < 0.:
        e2y = -e2y
    e2z = pca2.components_[2].copy()
    if e2z @ secondMoment2 < 0.:
        e2z = -e2z

    basis1 = [e1x, e1y, e1z]
    basis2 = [e2x, e2y, e2z]
    # print("basis1:", basis1)
    # print("basis2:", basis2)

    rotation = computeTransitionMatrix(basis1, basis2)
    # print("rotation:", rotation)

    center1_rotated = rotation @ pca1.mean_
    shift = pca2.mean_ - center1_rotated
    return (rotation, shift)

def map(v, rotation, shift):
    return rotation @ v + shift

def mapModel(model, rotation, shift):
    return np.array([map(v, rotation, shift) for v in model])

def computeTransitionMatrix(basis1, basis2):
    """Define orthogonal matrix that tarnsforms basis1 to basis2"""

    # Standard basis
    e = []
    e.append(np.array([1., 0., 0.]))
    e.append(np.array([0., 1., 0.]))
    e.append(np.array([0., 0., 1.]))

    # R3Matrix alpha, beta;
    alpha = np.array([[0.]*3, [0.]*3, [0.]*3])
    beta = np.array([[0.]*3, [0.]*3, [0.]*3])

    # Express basis vectors via basis1
    # e_i = \sum_i alpha_{ij} basis1_j
    alpha[0][0] = e[0] @ basis1[0];
    alpha[0][1] = e[0] @ basis1[1];
    alpha[0][2] = e[0] @ basis1[2];

    alpha[1][0] = e[1] @ basis1[0];
    alpha[1][1] = e[1] @ basis1[1];
    alpha[1][2] = e[1] @ basis1[2];

    alpha[2][0] = e[2] @ basis1[0];
    alpha[2][1] = e[2] @ basis1[1];
    alpha[2][2] = e[2] @ basis1[2];

    # Express basis2_i via standard basis vectors
    # v_j = \sum_k beta_{jk} e_k
    beta[0][0] = basis2[0] @ e[0];
    beta[0][1] = basis2[0] @ e[1];
    beta[0][2] = basis2[0] @ e[2];

    beta[1][0] = basis2[1] @ e[0];
    beta[1][1] = basis2[1] @ e[1];
    beta[1][2] = basis2[1] @ e[2];

    beta[2][0] = basis2[2] @ e[0];
    beta[2][1] = basis2[2] @ e[1];
    beta[2][2] = basis2[2] @ e[2];

    # m * e_i = \sum_j alpha_{ij} v_j  =
    #         = \sum_j alpha_{ij} \sum_k beta_{jk} e_k
    # Let gamma = alpha * beta
    # Rows of gamma are images of basis vectors
    # Copy them to columns of resulting matrix
    # R3Matrix gamma = alpha * beta;
    gamma = alpha @ beta

    m = np.array([[0.]*3, [0.]*3, [0.]*3])
    m[0][0] = gamma[0][0];
    m[1][0] = gamma[0][1];
    m[2][0] = gamma[0][2];

    m[0][1] = gamma[1][0];
    m[1][1] = gamma[1][1];
    m[2][1] = gamma[1][2];

    m[0][2] = gamma[2][0];
    m[1][2] = gamma[2][1];
    m[2][2] = gamma[2][2];
    return m
