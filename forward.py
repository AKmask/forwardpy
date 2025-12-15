# -*- coding: utf-8 -*-
"""
Copyright © 2023-2025 AKmask (Bohan Chen)
Created on 2024/10/01
uploading 2025/12/15
@Email: 1392009424@qq.com
@author: AKmask (Bohan Chen)
"""
from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import datetime

def get_vpml(velocity, nx, nz, pml):
    '''
    Add PML to the velocity model.
    ------------------------------------------
    :param velocity: (array) velocity model.
    :param nx: (int) X-axis length of velocity model.
    :param nz: (int) Z-axis length of velocity model.
    :param pml: (int) Thickness of the PML layer.
    :return: (array) A velocity model with PML layer.
    '''
    vpml = torch.zeros((nz + 2 * pml, nx + 2 * pml), dtype=torch.float32)
    vpml[pml:pml + nz, pml:pml + nx] = velocity
    #left
    vpml[pml:pml + nz, 0: pml] = velocity[:, 0:1]
    #right
    vpml[pml:pml + nz, pml + nx:2 * pml + nx] = velocity[:, nx - 1:nx]
    #up
    vpml[0:pml, :] = vpml[pml, :]
    #down
    vpml[pml + nz:2 * pml + nz, :] = vpml[pml + nz - 1, :]
    return vpml

def get_Propagation_coefficient(nx, nz, pml, Vp, Vs, Den):
    '''
    Lamé coefficient calculation.
    ------------------------------------------
    :param nx: (int) X-axis length of velocity model.
    :param nz: (int) Z-axis length of velocity model.
    :param pml: (int) Thickness of the PML layer.
    :param Vp: (array) P velocity model.
    :param Vs: (array) S velocity model.
    :param Den: (array) Density velocity model.
    :return: (array) Lamé coefficient.
    '''
    n = 2 * pml + nz
    m = 2 * pml + nx
    R2 = torch.zeros((n, m), dtype=torch.float32)
    R1 = torch.zeros((n, m), dtype=torch.float32)
    D = torch.zeros((n, m), dtype=torch.float32)
    # λ
    R2[0:n, 0:m] = Den * (Vp ** 2) - 2 * Den * (Vs ** 2)
    # μ
    R1[0:n, 0:m] = Den * (Vs ** 2)
    # λ+2μ
    D[0:n, 0:m] = R2 + 2 * R1

    return R2, R1, D

def theoretical_reflection_coefficient(nx, nz, pml, Vp_max, dx, dz):
    '''
    Add absorption values to the PML layer.
    ------------------------------------------
    :param nx: (int) X-axis length of velocity model.
    :param nz: (int) Z-axis length of velocity model.
    :param pml: (int) Thickness of the PML layer.
    :param Vp_max: (int) Maximum of P velocity model.
    :param dx: (int) X-direction spatial stride.
    :param dz: (int) Z-direction spatial stride.
    :return: (array) Absorption values.
    '''
    R = 1e-4 # attenuation factor
    n = 2 * pml + nz
    m = 2 * pml + nx
    plx = pml * dx;
    plz = pml * dz;
    ddx = torch.zeros((n, m), dtype=torch.float32)
    ddz = torch.zeros((n, m), dtype=torch.float32)
    for i in range(n):
        for k in range(m):
            # up-left
            if 0 <= i < pml and 0 <= k < pml:
                x = pml - k
                z = pml - i
                ddx[i, k] = -np.log(R) * 3 * Vp_max * x ** 2 / (2 * plx ** 2)
            # up-right
            elif 0 <= i < pml and m - pml <= k < m:
                x = k - (m - pml)
                z = pml - i
                ddx[i, k] = -np.log(R) * 3 * Vp_max * x ** 2 / (2 * plx ** 2)
                ddz[i, k] = -np.log(R) * 3 * Vp_max * z ** 2 / (2 * plz ** 2)
            # down-left
            elif n - pml <= i < n and 0 <= k < pml:
                x = pml - k
                z = i - (n - pml)
                ddx[i, k] = -np.log(R) * 3 * Vp_max * x ** 2 / (2 * plx ** 2)
                ddz[i, k] = -np.log(R) * 3 * Vp_max * z ** 2 / (2 * plz ** 2)
            # down-right
            elif n - pml <= i < n and m - pml <= k < m:
                x = k - (m - pml)
                z = i - (n - pml)
                ddx[i, k] = -np.log(R) * 3 * Vp_max * x ** 2 / (2 * plx ** 2)
                ddz[i, k] = -np.log(R) * 3 * Vp_max * z ** 2 / (2 * plz ** 2)
            # up-middle
            elif 0 <= i < pml and pml <= k < m - pml + 1:
                x = 0
                z = pml - i
                ddx[i, k] = 0
                ddz[i, k] = -np.log(R) * 3 * Vp_max * z ** 2 / (2 * plz ** 2)
            # down-middle
            elif n - pml <= i < n and pml <= k < m - pml:
                x = 0
                z = i - (n - pml)
                ddx[i, k] = 0
                ddz[i, k] = -np.log(R) * 3 * Vp_max * z ** 2 / (2 * plz ** 2)
            # left-middle
            elif pml <= i < n - pml and 0 <= k < pml:
                x = pml - k
                z = 0
                ddx[i, k] = -np.log(R) * 3 * Vp_max * x ** 2 / (2 * plx ** 2)
                ddz[i, k] = 0
            # right-middle
            elif pml <= i < n - pml and m - pml <= k < m:
                x = k - (m - pml)
                z = 0
                ddx[i, k] = -np.log(R) * 3 * Vp_max * x ** 2 / (2 * plx ** 2)
                ddz[i, k] = 0
    ddx = torch.tensor(ddx, dtype=torch.float32)
    ddz = torch.tensor(ddz, dtype=torch.float32)
    return ddx, ddz

def make_ricker2(lt):
    '''
    Generate ricker wave energy.
    ------------------------------------------
    :param lt: (int) Sampling time.
    :return: (array) Ricker wave energy.
    '''
    ricker = torch.zeros(lt, dtype=torch.float32)
    fm = torch.tensor(40.0, dtype=torch.float32) # frequency
    dt = torch.tensor(0.0001, dtype=torch.float32) # time step
    A = torch.tensor(1.0, dtype=torch.float32) # amplitude
    tt = torch.arange(-0.02, 0.02 + dt, dt, dtype=torch.float32)
    ricker = A * (1 - 2 * (np.pi * fm * tt) ** 2) * np.exp(-(np.pi * fm * tt) ** 2)
    return ricker

def difference(pic, pic2):
    if torch.is_tensor(pic):
        if pic.is_cuda:
            pic = pic.cpu().detach().numpy()
        else:
            pic = pic.detach().numpy()
    if torch.is_tensor(pic2):
        if pic2.is_cuda:
            pic2 = pic2.cpu().detach().numpy()
        else:
            pic2 = pic2.detach().numpy()
    vmin, vmax = pic2.min(), pic2.max()
    fig, axes = plt.subplots(3, 1, figsize=(18, 9))
    #fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    im0 = axes[0].imshow(pic, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title("conv")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(pic2, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title("slice")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(pic - pic2, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axes[2].set_title("conv - slice")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes.flat:
        ax.axis("on")
    plt.tight_layout()
    plt.show()

def make_trace_data_1_pic(CONV_wave, FDTD_wave):
    '''
    if torch.is_tensor(CONV_wave) and CONV_wave.is_cuda:
        CONV = CONV_wave.cpu().detach().numpy()
        FDTD = FDTD_wave.cpu().detach().numpy()
    '''
    if torch.is_tensor(CONV_wave) and CONV_wave.is_cuda:
        CONV_wave = CONV_wave.cpu().detach().numpy()
        FDTD_wave = FDTD_wave.cpu().detach().numpy()

    CONV = CONV_wave
    FDTD = FDTD_wave
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=120, sharex=False)
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    ax1.plot(CONV, 'b-', linewidth=0.8, label='CONV')
    ax1.plot(FDTD, 'r--', linewidth=0.8, label='FDTD')
    y_min = min(np.min(CONV), np.min(FDTD))
    y_max = max(np.max(CONV), np.max(FDTD))
    ax1.set_ylim(y_min * 1.1, y_max * 1.1)
    ax1.set_ylabel('Amplitude(Pa)', fontsize=12)
    # ax1.set_title('Single-trace Comparison: Numerical Dispersion Analysis', fontsize=14)
    ax1.legend(loc='upper left', frameon=False, shadow=False)
    difference = CONV - FDTD
    difference -= np.mean(difference)
    ax2.plot(difference, 'g--', linewidth=0.8)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    margin = max(abs(y_min), abs(y_max)) * 0.1
    ax2.set_ylim(y_min - margin, y_max + margin)
    #ax2.set_xlabel('Time Step (sample index)', fontsize=12)
    ax2.set_ylabel('Amplitude(Pa)', fontsize=12)
    ax2.legend(['Difference'], loc='upper left', frameon=False, shadow=False)
    ax1.grid(False)
    ax2.grid(False)
    plt.tight_layout()
    plt.show()

def tradi6(lt,  cuda_use=True):
    '''
    A method for implementing finite difference calculations using matrix slicing operations.
    ------------------------------------------
    :param lt: (int) Sampling time.
    :param cuda_use: (bool) GPU using or not.
    :return: P and S wavefield snapshot at the corresponding sampling time.
    '''
    if cuda_use:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dz = 5 # x-direction spatial stride
    dx = 5 # z-direction spatial stride
    # load P/S velocity models and density model
    velocity_p = np.array(loadmat("Vp_magnify1000_resize_down10.mat")['Data'])
    velocity_p_torch = torch.from_numpy(velocity_p).float().to(device)
    velocity_s = np.array(loadmat("Vs_magnify1000_resize_down10.mat")['Data'])
    velocity_s_torch = torch.from_numpy(velocity_s).float().to(device)
    velocity_den = np.array(loadmat("density_magnify1000_resize_down10.mat")['Data'])
    velocity_den_torch = torch.from_numpy(velocity_den).float().to(device)
    Nx = velocity_p.shape[1] # length of velocity model
    Nz = velocity_p.shape[0] # height of velocity model
    pml = 100 # thickness of PML layer
    n = Nz + 2 * pml
    m = Nx + 2 * pml
    dt = 0.0001 # time step
    # add PML layer
    Vp = get_vpml(velocity_p_torch, Nx, Nz, pml)
    Vp_max = torch.max(torch.max(Vp))
    Vs = get_vpml(velocity_s_torch, Nx, Nz, pml)
    d = get_vpml(velocity_den_torch, Nx, Nz, pml)
    R2, R1, D = get_Propagation_coefficient(Nx, Nz, pml, Vp, Vs, d) # get lamé coefficient
    ricker = make_ricker2(lt) # get ricker wave energy
    source = [(680,100 )] # hypocentral location
    ddx, ddz = theoretical_reflection_coefficient(Nx, Nz, pml, Vp_max, dx, dz) # get absorption values to the PML layer
    if cuda_use:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Vp = Vp.to(device)
        Vs = Vs.to(device)
        d = d.to(device)
        R2 = R2.to(device)
        R1 = R1.to(device)
        D = D.to(device)
        ricker = ricker.to(device)
        ddx = ddx.to(device)
        ddz = ddz.to(device)
        dt = torch.tensor(dt, dtype=torch.float32).to(device)

    sx = source[0][0]
    sz = source[0][1]
    x0 = pml + sx
    z0 = pml + sz
    if cuda_use:
        Pz = torch.zeros((Nz, Nx), dtype=torch.float32).to(device)
        Px = torch.zeros((Nz, Nx), dtype=torch.float32).to(device)
        pxx = torch.zeros((n, m), dtype=torch.float32).to(device) # x-direction normal stress field
        pzz = torch.zeros((n, m), dtype=torch.float32).to(device) # z-direction normal stress field
        pxz = torch.zeros((n, m), dtype=torch.float32).to(device) # shear stress field
        Vx = torch.zeros((n, m), dtype=torch.float32).to(device) # s-wave velocity field
        Vz = torch.zeros((n, m), dtype=torch.float32).to(device) # p-wave velocity field
        pxx1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        pzz1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        pxz1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Vx1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Vz1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        pxx2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        pzz2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        pxz2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Vx2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Vz2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        shot_record_Vz = torch.zeros((lt, m), dtype=torch.float32).to(device) # seismic data recording
        shot_record_Vx = torch.zeros((lt, m), dtype=torch.float32).to(device)
        receive_Vz = torch.zeros(lt, dtype=torch.float32).to(device)
        receive_Vx = torch.zeros(lt, dtype=torch.float32).to(device)
    else:
        Pz = np.zeros((Nz, Nx), dtype=np.float32)
        Px = np.zeros((Nz, Nx), dtype=np.float32)
        pxx = np.zeros((n, m), dtype=np.float32)
        pzz = np.zeros((n, m), dtype=np.float32)
        pxz = np.zeros((n, m), dtype=np.float32)
        Vx = np.zeros((n, m), dtype=np.float32)
        Vz = np.zeros((n, m), dtype=np.float32)
        pxx1 = np.zeros((n, m), dtype=np.float32)
        pzz1 = np.zeros((n, m), dtype=np.float32)
        pxz1 = np.zeros((n, m), dtype=np.float32)
        Vx1 = np.zeros((n, m), dtype=np.float32)
        Vz1 = np.zeros((n, m), dtype=np.float32)
        pxx2 = np.zeros((n, m), dtype=np.float32)
        pzz2 = np.zeros((n, m), dtype=np.float32)
        pxz2 = np.zeros((n, m), dtype=np.float32)
        Vx2 = np.zeros((n, m), dtype=np.float32)
        Vz2 = np.zeros((n, m), dtype=np.float32)
        shot_record = np.zeros((lt, Nx), dtype=np.float32)
    k = slice(3, m - 4)
    i = slice(3, n - 4)
    d_dx = d[i,k] * dx
    d_dz = d[i,k] * dz
    D_dt_dz = D[i,k] * dt / dz
    D_dt_dx = D[i,k] * dt / dx

    R2_dt_dx = R2[i,k] * dt / dx
    R2_dt_dz = R2[i,k] * dt / dz

    R1_dt_dx = R1[i,k] * dt / dx
    R1_dt_dz = R1[i,k] * dt / dz
    start = datetime.datetime.now() # time runs start
    for it in range(lt):
        # injecting ricker wavelet energy
        if it < 400:
            pzz[z0, x0] = pzz[z0, x0] + ricker[it]
            pxx[z0, x0] = pxx[z0, x0] + ricker[it]

        # calculation of p-wave velocity field
        Vz1[i, k] = ((1 - 0.5 * dt * ddz[i.start + 1:i.stop + 1, k]) * Vz1[i, k] +
                     dt * ((pzz[i.start + 1:i.stop + 1, k] - pzz[i, k]) * 1.171875 -
                           (pzz[i.start + 2:i.stop + 2, k] - pzz[i.start - 1:i.stop - 1, k]) * 0.065104166666667 +
                           (pzz[i.start + 3:i.stop + 3, k] - pzz[i.start - 2:i.stop - 2, k]) * 0.0046875) /
                     (d_dz)) / (1 + 0.5 * dt * ddz[i.start - 1:i.stop - 1, k])

        Vz2[i, k] = ((1 - 0.5 * dt * ddx[i, k.start - 1:k.stop - 1]) * Vz2[i, k] +
                     dt * ((pxz[i, k] - pxz[i, k.start - 1:k.stop - 1]) * 1.171875 -
                           (pxz[i, k.start + 1:k.stop + 1] - pxz[i, k.start - 2:k.stop - 2]) * 0.065104166666667 +
                           (pxz[i, k.start + 2:k.stop + 2] - pxz[i, k.start - 3:k.stop - 3]) * 0.0046875) /
                     (d_dx)) / (1 + 0.5 * dt * ddx[i, k.start + 1:k.stop + 1])

        Vz[i, k] = Vz1[i, k] + Vz2[i, k]
        # calculation of s-wave velocity field
        Vx1[i, k] = ((1 - 0.5 * dt * ddx[i, k.start + 2:k.stop + 2]) * Vx1[i, k] +
                     dt * ((pxx[i, k.start + 1:k.stop + 1] - pxx[i, k]) * 1.171875 -
                           (pxx[i, k.start + 2:k.stop + 2] - pxx[i, k.start - 1:k.stop - 1]) * 0.065104166666667 +
                           (pxx[i, k.start + 3:k.stop + 3] - pxx[i, k.start - 2:k.stop - 2]) * 0.0046875) /
                     (d_dx)) / (1 + 0.5 * dt * ddx[i, k.start - 2:k.stop - 2])

        Vx2[i, k] = ((1 - 0.5 * dt * ddz[i.start + 2:i.stop + 2, k]) * Vx2[i, k] +
                     dt * ((pxz[i, k] - pxz[i.start - 1:i.stop - 1, k]) * 1.171875 -
                           (pxz[i.start + 1:i.stop + 1, k] - pxz[i.start - 2:i.stop - 2, k]) * 0.065104166666667 +
                           (pxz[i.start + 2:i.stop + 2, k] - pxz[i.start - 3:i.stop - 3, k]) * 0.0046875) /
                     (d_dz)) / (1 + 0.5 * dt * ddz[i.start - 2:i.stop - 2, k])

        Vx[i, k] = Vx1[i, k] + Vx2[i, k]
        # calculation of z-direction normal stress field
        pzz1[i, k] = ((1 - 0.5 * dt * ddz[i, k.start - 1:k.stop - 1]) * pzz1[i, k] +
                      D_dt_dz * ((Vz[i, k] - Vz[i.start - 1:i.stop - 1, k]) * 1.171875 -
                                      (Vz[i.start + 1:i.stop + 1, k] - Vz[i.start - 2:i.stop - 2,k]) * 0.065104166666667 +
                                      (Vz[i.start + 2:i.stop + 2, k] - Vz[i.start - 3:i.stop - 3, k]) * 0.0046875)
                      ) / (1 + 0.5 * dt * ddz[i, k.start + 1:k.stop + 1])

        pzz2[i, k] = ((1 - 0.5 * dt * ddx[i.start + 1:i.stop + 1, k]) * pzz2[i, k] +
                      R2_dt_dx * ((Vx[i, k] - Vx[i, k.start - 1:k.stop - 1]) * 1.171875 -
                                       (Vx[i, k.start + 1:k.stop + 1] - Vx[i,
                                                                        k.start - 2:k.stop - 2]) * 0.065104166666667 +
                                       (Vx[i, k.start + 2:k.stop + 2] - Vx[i, k.start - 3:k.stop - 3]) * 0.0046875)
                      ) / (1 + 0.5 * dt * ddx[i.start - 1:i.stop - 1, k])

        pzz[i, k] = pzz1[i, k] + pzz2[i, k]
        # calculation of x-direction normal stress field
        pxx1[i, k] = ((1 - 0.5 * dt * ddx[i.start + 2:i.stop + 2, k]) * pxx1[i, k] +
                      D_dt_dx * ((Vx[i, k] - Vx[i, k.start - 1:k.stop - 1]) * 1.171875 -
                                      (Vx[i, k.start + 1:k.stop + 1] - Vx[i,
                                                                       k.start - 2:k.stop - 2]) * 0.065104166666667 +
                                      (Vx[i, k.start + 2:k.stop + 2] - Vx[i, k.start - 3:k.stop - 3]) * 0.0046875)
                      ) / (1 + 0.5 * dt * ddx[i.start - 2:i.stop - 2, k])

        pxx2[i, k] = ((1 - 0.5 * dt * ddz[i, k.start - 2:k.stop - 2]) * pxx2[i, k] +
                      R2_dt_dz * ((Vz[i, k] - Vz[i.start - 1:i.stop - 1, k]) * 1.171875 -
                                       (Vz[i.start + 1:i.stop + 1, k] - Vz[i.start - 2:i.stop - 2,
                                                                        k]) * 0.065104166666667 +
                                       (Vz[i.start + 2:i.stop + 2, k] - Vz[i.start - 3:i.stop - 3, k]) * 0.0046875)
                      ) / (1 + 0.5 * dt * ddz[i, k.start + 2:k.stop + 2])

        pxx[i, k] = pxx1[i, k] + pxx2[i, k]
        # calculation of shear stress field
        pxz1[i, k] = ((1 - 0.5 * dt * ddx[i.start + 1:i.stop + 1, k.start + 1:k.stop + 1]) * pxz1[i, k] +
                      R1_dt_dx * ((Vz[i, k.start + 1:k.stop + 1] - Vz[i, k]) * 1.171875 -
                                       (Vz[i, k.start + 2:k.stop + 2] - Vz[i,
                                                                        k.start - 1:k.stop - 1]) * 0.065104166666667 +
                                       (Vz[i, k.start + 3:k.stop + 3] - Vz[i, k.start - 2:k.stop - 2]) * 0.0046875)
                      ) / (1 + 0.5 * dt * ddx[i, k])

        pxz2[i, k] = ((1 - 0.5 * dt * ddz[i.start + 1:i.stop + 1, k.start + 1:k.stop + 1]) * pxz2[i, k] +
                      R1_dt_dz * ((Vx[i.start + 1:i.stop + 1, k] - Vx[i, k]) * 1.171875 -
                                       (Vx[i.start + 2:i.stop + 2, k] - Vx[i.start - 1:i.stop - 1,
                                                                        k]) * 0.065104166666667 +
                                       (Vx[i.start + 3:i.stop + 3, k] - Vx[i.start - 2:i.stop - 2, k]) * 0.0046875)
                      ) / (1 + 0.5 * dt * ddz[i, k])
        pxz[i, k] = pxz1[i, k] + pxz2[i, k]
        shot_record_Vz[it, :] = Vz[120, :]
        shot_record_Vx[it, :] = Vx[120, :]
        receive_Vz[it] = Vz[200, 750]
        receive_Vx[it] = Vx[200, 750]
    end = datetime.datetime.now()
    print("Time",end - start)
    print("6_order_FDTD_end")
    return Vz,Vx,shot_record_Vz,shot_record_Vx,receive_Vz,receive_Vx

def conv6(lt,  cuda_use=True):
    '''
    Using convolution to implement finite difference calculations.
    ------------------------------------------
    :param lt: (int) Sampling time.
    :param cuda_use: (bool) GPU using or not.
    :return: P and S wavefield snapshot at the corresponding sampling time.
    '''
    if cuda_use:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    velocity_p = np.array(loadmat("Vp_magnify1000_resize_down10.mat")['Data'])
    velocity_p_torch = torch.from_numpy(velocity_p).float().to(device)
    velocity_s = np.array(loadmat("Vs_magnify1000_resize_down10.mat")['Data'])
    velocity_s_torch = torch.from_numpy(velocity_s).float().to(device)
    velocity_den = np.array(loadmat("density_magnify1000_resize_down10.mat")['Data'])
    velocity_den_torch = torch.from_numpy(velocity_den).float().to(device)

    dt = 0.0001
    dx = 5
    dz = 5
    Nx = velocity_p.shape[1]
    Nz = velocity_p.shape[0]
    pml = 100
    n = Nz + 2 * pml
    m = Nx + 2 * pml
    Vp = get_vpml(velocity_p_torch, Nx, Nz, pml)
    Vp_max = torch.max(torch.max(Vp))
    Vs = get_vpml(velocity_s_torch, Nx, Nz, pml)
    d = get_vpml(velocity_den_torch, Nx, Nz, pml)
    R2, R1, D = get_Propagation_coefficient(Nx, Nz, pml, Vp, Vs, d)
    ddx, ddz = theoretical_reflection_coefficient(Nx, Nz, pml, Vp_max, dx, dz)
    ricker = make_ricker2(lt)

    source = [(680,100 )]
    sx = source[0][0]
    sz = source[0][1]
    x0 = pml + sx
    z0 = pml + sz
    #convolution kernel
    kernel_x = torch.tensor([[[[-0.0046875, 0.065104166666667, -1.171875, 1.171875, -0.065104166666667, 0.0046875]]]],
                            dtype=torch.float32)
    if cuda_use:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Tzz = torch.zeros((n, m), dtype=torch.float32).to(device) # z-direction normal stress field
        Tzz1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Tzz2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Txx = torch.zeros((n, m), dtype=torch.float32).to(device) # x-direction normal stress field
        Txx1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Txx2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Txz = torch.zeros((n, m), dtype=torch.float32).to(device) # shear stress field
        Txz1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Txz2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Vx = torch.zeros((n, m), dtype=torch.float32).to(device) # s-wave velocity field
        Vx1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Vx2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Vz = torch.zeros((n, m), dtype=torch.float32).to(device) # p-wave velocity field
        Vz1 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Vz2 = torch.zeros((n, m), dtype=torch.float32).to(device)
        Pz = torch.zeros((Nz, Nx), dtype=torch.float32).to(device)
        Px = torch.zeros((Nz, Nx), dtype=torch.float32).to(device)

        shot_record_Vz = torch.zeros((lt, m), dtype=torch.float32).to(device)  # seismic data recording
        shot_record_Vx = torch.zeros((lt, m), dtype=torch.float32).to(device)
        receive_Vz = torch.zeros(lt, dtype=torch.float32).to(device)
        receive_Vx = torch.zeros(lt, dtype=torch.float32).to(device)

    if cuda_use:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Vp = Vp.to(device)
        Vs = Vs.to(device)
        d = d.to(device)
        R2 = R2.to(device)
        R1 = R1.to(device)
        D = D.to(device)
        ricker = ricker.to(device)
        ddx = ddx.to(device)
        ddz = ddz.to(device)
        dt = torch.tensor(dt, dtype=torch.float32).to(device)
        kernel_x = kernel_x.to(device)
        ricker = ricker.to(device)
        #shot_record = shot_record.to(device)
    #Calculation of attenuation parameters
    half_dt_ddx = 0.5 * dt * ddx
    half_dt_ddz = 0.5 * dt * ddz

    dt_d_dx = dt / (d * dx)
    dt_d_dz = dt / (d * dz)

    D_dt_dz = D * dt / dz
    D_dt_dx = D * dt / dx

    R2_dt_dx = R2 * dt / dx
    R2_dt_dz = R2 * dt / dz

    R1_dt_dx = R1 * dt / dx
    R1_dt_dz = R1 * dt / dz

    start = datetime.datetime.now() # time runs strat
    for it in range(lt):
        if it < 400:
            Tzz[z0, x0] = Tzz[z0, x0] + ricker[it]
            Txx[z0, x0] = Txx[z0, x0] + ricker[it]
        # calculation of s-wave velocity field
        dz_Txz = F.pad(
            F.conv2d(Txz[0:n, 1:m-1].t().unsqueeze(0).unsqueeze(0), kernel_x, stride=1, padding=0), pad=(3, 2, 1, 1), value = 0).squeeze(
            0).squeeze(0)
        Vx1 = ((1 - half_dt_ddx) * Vx1 +
               dt_d_dx * F.pad(
                    F.conv2d(Txx[1:n, 1:m].unsqueeze(0).unsqueeze(0), kernel_x, stride=1, padding=0), pad=(3, 3, 1, 0), value = 0).squeeze(
                        0).squeeze(0)) / (1 + half_dt_ddx)
        Vx2 = ((1 - half_dt_ddz) * Vx2 + dt_d_dz * dz_Txz.t()) / (1 + half_dt_ddz)
        Vx = Vx1 + Vx2
        # calculation of p-wave velocity field
        dx_Txz = F.pad(
            F.conv2d(Txz[1:n-1, 0:m].unsqueeze(0).unsqueeze(0), kernel_x, stride=1, padding=0),pad=(3, 2, 1, 1), value = 0).squeeze(
            0).squeeze(0)
        dz_Tzz = dt_d_dz * F.pad(
            F.conv2d(Tzz[1:n, 1:m].t().unsqueeze(0).unsqueeze(0), kernel_x, stride=1, padding=0), pad=(3, 3, 1, 0), value = 0).squeeze(
            0).squeeze(
                0).t()
        Vz1 = ((1 - half_dt_ddz) * Vz1 + dz_Tzz) / (1 + half_dt_ddz)
        Vz2 = ((1 - half_dt_ddx) * Vz2 + dt_d_dx * dx_Txz) / (1 + half_dt_ddx)
        Vz = Vz1 + Vz2
        # calculation of x-direction normal stress field
        dx_Vx = F.pad(
            F.conv2d(Vx[1:n, 0:m].unsqueeze(0).unsqueeze(0), kernel_x, stride=1, padding=0),
            pad=(3, 2, 1, 0), value = 0).squeeze(0).squeeze(0)
        dz_Vz = F.pad(
            F.conv2d(Vz[0:n, 1:m].t().unsqueeze(0).unsqueeze(0), kernel_x, stride=1, padding=0), pad=(3, 2, 1, 0), value = 0).squeeze(0).squeeze(
                0)
        Txx1 = ((1 - half_dt_ddx) * Txx1 + D_dt_dx * dx_Vx) / (1 + half_dt_ddx)
        Txx2 = ((1 - half_dt_ddz) * Txx2 + R2_dt_dz * dz_Vz.t()) / (1 + half_dt_ddz)
        Txx = Txx1 + Txx2
        # calculation of z-direction normal stress field
        Tzz1 = ((1 - half_dt_ddz) * Tzz1 + D_dt_dz * dz_Vz.t()) / (1 + half_dt_ddz)
        Tzz2 = ((1 - half_dt_ddx) * Tzz2 + R2_dt_dx * dx_Vx) / (1 + half_dt_ddx)
        Tzz = Tzz1 + Tzz2
        # calculation of shear stress field
        Txz1 = ((1 - half_dt_ddx) * Txz1 +
                R1_dt_dx * F.pad(
                    F.conv2d(Vz[1:n-1, 1:m].unsqueeze(0).unsqueeze(0), kernel_x, stride=1, padding=0), pad=(3, 3, 1, 1), value = 0).squeeze(
                        0).squeeze(0)) / (1 + half_dt_ddx)
        Txz2 = ((1 - half_dt_ddz) * Txz2 +
                R1_dt_dz * F.pad(
                    F.conv2d(Vx[1:n, 1:m-1].t().unsqueeze(0).unsqueeze(0), kernel_x, stride=1, padding=0), pad=(3, 3, 1, 1), value = 0).squeeze(
                        0).squeeze(0).t()) / (1 + half_dt_ddz)
        Txz = Txz1 + Txz2
        shot_record_Vz[it, :] = Vz[120, :]
        shot_record_Vx[it, :] = Vx[120, :]
        receive_Vz[it] = Vz[200, 750]
        receive_Vx[it] = Vx[200, 750]
    end = datetime.datetime.now() # time runs out
    print("Time：", end - start)
    print("6_order_conv_end")
    return Vz,Vx,shot_record_Vz,shot_record_Vx,receive_Vz,receive_Vx


if __name__ == '__main__':
    '''
        This code compares the performance of the convolution method and the finite difference method under 
    the time resolution of 2nd order and the spatial resolution of 6th order.
        The output results include a comparison of the computational speed of the two methods and snapshots of the p-wave and s-wave fields.
    '''
    # convolution method
    pz61,px61,shot_record_Vz1,shot_record_Vx1,receive_Vz1,receive_Vx1 = conv6( 4000,  cuda_use=True)
    # FDTD method
    pz62, px62,shot_record_Vz2,shot_record_Vx2,receive_Vz2,receive_Vx2 =tradi6( 4000,  cuda_use=True)
    print("compara P wave")
    # p-wave field imaging comparison
    difference(pz61, pz62)
    print("compara S wave")
    # s-wave field imaging comparison
    difference(px61, px62)
    print("compara P receive")
    make_trace_data_1_pic(receive_Vz1, receive_Vz2)
    print("compara S receive")
    make_trace_data_1_pic(receive_Vx1, receive_Vx2)



