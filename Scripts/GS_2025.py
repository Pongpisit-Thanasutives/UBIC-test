import os
import sys; sys.path.append('../')
import numpy as np
from numpy.random import default_rng

import scipy.io as sio
import pysindy as ps
from PDE_FIND import build_linear_system, print_pde, TrainSTRidge, measure_pce
from best_subset import *
from frols import frols
from UBIC import *
from solvel0 import solvel0
from findiff import FinDiff
import sgolay2

n = 128
data = sio.loadmat(f"../Datasets/Big/reaction_diffusion_3d_{n}.mat")
u_sol = np.array((data['usol']).real, dtype=np.float32)
v_sol = np.array((data['vsol']).real, dtype=np.float32)
x = np.array((data['x'][0]).real, dtype=np.float32)
y = np.array((data['y'][0]).real, dtype=np.float32)
z = np.array((data['z'][0]).real, dtype=np.float32)
t = np.array((data['t'][0]).real, dtype=np.float32)

del data

dt = t[1] - t[0]
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

u = np.zeros((n, n, n, len(t), 2))
u[:, :, :, :, 0] = u_sol
u[:, :, :, :, 1] = v_sol
del u_sol, v_sol

# Add noise
np.random.seed(100)
noise_lv = 0.1
domain_noise = 0.01*np.abs(noise_lv)*np.std(u)*np.random.randn(*u.shape)
u = u + domain_noise

denoise = True
if denoise:
    un = u[:, :, :, :, 0]
    vn = u[:, :, :, :, 1]

    div = 30
    ws = max(un.shape[:-2])//div; po = 5
    if ws%2 == 0: ws -=1

    nun = np.zeros_like(un)
    for i in trange(un.shape[-1]):
        for j in range(un.shape[-2]):
            nun[:, :, j, i] = sgolay2.SGolayFilter2(window_size=ws, poly_order=po)(un[:, :, j, i])
    un = nun.copy()
    del nun

    nvn = np.zeros_like(vn)
    for i in trange(vn.shape[-1]):
        for j in range(vn.shape[-2]):
            nvn[:, :, j, i] = sgolay2.SGolayFilter2(window_size=ws, poly_order=po)(vn[:, :, j, i])
    vn = nvn.copy()
    del nvn

    dim = 10

    un = un.reshape(-1, len(t))
    uun, sigmaun, vun = np.linalg.svd(un, full_matrices=False); vun = vun.T
    un = uun[:,0: dim].dot(np.diag(sigmaun[0:dim]).dot(vun[:,0:dim].T))
    un = un.reshape(len(x), len(y), len(z), len(t))

    vn = vn.reshape(-1, len(t))
    uvn, sigmavn, vvn = np.linalg.svd(vn, full_matrices=False); vvn = vvn.T
    vn = uvn[:,0: dim].dot(np.diag(sigmavn[0:dim]).dot(vvn[:,0:dim].T))
    vn = vn.reshape(len(x), len(y), len(z), len(t))

    u = np.stack([un, vn], axis=-1)
    del un, vn, uun, uvn

# Need to define the 2D spatial grid before calling the library
# X, Y, Z, T = np.meshgrid(x, y, z, t, indexing="ij")
spatiotemporal_grid = np.asarray(np.meshgrid(x, y, z, t, indexing="ij"))
spatiotemporal_grid = np.transpose(spatiotemporal_grid, axes=[1, 2, 3, 4, 0])
weak_lib = ps.WeakPDELibrary(
    function_library=ps.PolynomialLibrary(degree=3,include_bias=False),
    derivative_order=2,
    spatiotemporal_grid=spatiotemporal_grid,
    is_uniform=True,
    include_interaction=False,
    include_bias=True,
    periodic=True,
    K=10000,
)

X_pre = weak_lib.fit_transform(u)
y_pre = weak_lib.convert_u_dot_integral(u)

np.save("X_pre_GS_2025.npy", X_pre)
np.save("y_pre_GS_2025.npy", y_pre)

