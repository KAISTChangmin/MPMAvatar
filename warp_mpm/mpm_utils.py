import warp as wp
from mpm_data_structure import *
import numpy as np
import math


# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)


@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    """
    B = F * wp.transpose(F)
    dev(B) = B - (1/3) * tr(B) * I

    For a compressible Rivlin neo-Hookean materia, the cauchy stress is given by:
    mu * J^(-2/3) * dev(B) + lam * J (J - 1) * I
    see: https://en.wikipedia.org/wiki/Neo-Hookean_solid
    """

    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )

    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def inverse_lower_triangle(
    M: wp.mat33
):
    M11 = M[0,0]
    M21 = M[1,0]
    M22 = M[1,1]
    M31 = M[2,0]
    M32 = M[2,1]
    M33 = M[2,2]
    invdet = 1.0 / (M11 * M22 * M33)

    return invdet * wp.mat33(M22*M33, 0.0, 0.0, -M21*M33, M11*M33, 0.0, M21*M32-M31*M22, -M11*M32, M11*M22)

@wp.func
def kirchoff_stress_Anisotropy(
    R_inv: wp.vec3, d: wp.mat33, face: wp.vec3, vol: float, state: MPMStateStruct, mu: float, lam: float, gamma: float, kappa: float
):
    iD11 = R_inv[0]
    iD12 = R_inv[1]
    iD22 = R_inv[2]

    Q_0 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    R_0 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wp.qr3(d, Q_0, R_0)
    if R_0[0,0] < 0:
        Q_1 = wp.mat33(-Q_0[0,0], Q_0[0,1], -Q_0[0,2], -Q_0[1,0], Q_0[1,1], -Q_0[1,2], -Q_0[2,0], Q_0[2,1], -Q_0[2,2])
        R_1 = wp.mat33(-R_0[0,0], -R_0[0,1], -R_0[0,2], 0.0, R_0[1,1], R_0[1,2], 0.0, 0.0, -R_0[2,2])
    else:
        Q_1 = Q_0
        R_1 = R_0
    if R_1[1,1] < 0:
        Q = wp.mat33(Q_1[0,0], -Q_1[0,1], -Q_1[0,2], Q_1[1,0], -Q_1[1,1], -Q_1[1,2], Q_1[2,0], -Q_1[2,1], -Q_1[2,2])
        R = wp.mat33(R_1[0,0], R_1[0,1], R_1[0,2], 0.0, -R_1[1,1], -R_1[1,2], 0.0, 0.0, -R_1[2,2])
    else:
        Q = Q_1
        R = R_1

    F11 = R[0,0] * iD11
    F12 = R[0,0] * iD12 + R[0,1] * iD22
    F22 = R[1,1] * iD22
    F2 = wp.mat22(F11, F12, 0.0, F22)
    
    RiDT = wp.mat33(F11, 0.0, 0.0, F12, F22, 0.0, R[0,2], R[1,2], R[2,2])
    iFTJ = wp.mat22(F22, 0.0, -F12, F11)

    U3 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V3 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig3 = wp.vec3(0.0)
    F3 = wp.mat33(F11, F12, 0.0, 0.0, F22, 0.0, 0.0, 0.0, 0.0)
    wp.svd3(F3, U3, sig3, V3)
    U = wp.mat22(U3[0,0], U3[0,1], U3[1,0], U3[1,1])
    V = wp.mat22(V3[0,0], V3[0,1], V3[1,0], V3[1,1])

    Rot = U * wp.transpose(V)
    J = F11 * F22

    K2 = 2.0 * mu * (F2 - Rot) + lam * (J - 1.0) * iFTJ

    dr11 = K2[0,0]
    dr12 = K2[0,1]
    dr22 = K2[1,1]
    dr13 = gamma * R[0,2]
    dr23 = gamma * R[1,2]
    if R[2,2] > 1.0:
        dr33 = 0.0
    else:
        dr33 = -kappa * (1.0 - R[2,2]) * (1.0 - R[2,2])

    dr = wp.mat33(dr11, dr12, dr13, 0.0, dr22, dr23, 0.0, 0.0, dr33)
    K3 = dr * RiDT
    K3_sym = wp.mat33(K3[0,0], K3[0,1], K3[0,2], K3[0,1], K3[1,1], K3[1,2], K3[0,2], K3[1,2], K3[2,2])

    RiDT_inv = inverse_lower_triangle(RiDT)
    P = Q * K3_sym * RiDT_inv
    P1 = wp.vec3(P[0,0], P[1,0], P[2,0])
    P2 = wp.vec3(P[0,1], P[1,1], P[2,1])
    P3 = wp.vec3(P[0,2], P[1,2], P[2,2])

    d3 = wp.vec3(d[0,2], d[1,2], d[2,2])

    f2 = -vol * (iD11 * P1 + iD12 * P2)
    f3 = -vol * iD22 * P2
    f1 = -(f2 + f3)

    v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
    wp.atomic_add(state.vertex_force, v1, f1)
    wp.atomic_add(state.vertex_force, v2, f2)
    wp.atomic_add(state.vertex_force, v3, f3)
    
    return vol * wp.outer(P3, d3)

@wp.func
def anisotropy_return_mapping(d: wp.mat33, model: MPMModelStruct, p: int):
    Q_0 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    R_0 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wp.qr3(d, Q_0, R_0)
    if R_0[0,0] < 0:
        Q_1 = wp.mat33(-Q_0[0,0], Q_0[0,1], -Q_0[0,2], -Q_0[1,0], Q_0[1,1], -Q_0[1,2], -Q_0[2,0], Q_0[2,1], -Q_0[2,2])
        R_1 = wp.mat33(-R_0[0,0], -R_0[0,1], -R_0[0,2], 0.0, R_0[1,1], R_0[1,2], 0.0, 0.0, -R_0[2,2])
    else:
        Q_1 = Q_0
        R_1 = R_0
    if R_1[1,1] < 0:
        Q = wp.mat33(Q_1[0,0], -Q_1[0,1], -Q_1[0,2], Q_1[1,0], -Q_1[1,1], -Q_1[1,2], Q_1[2,0], -Q_1[2,1], -Q_1[2,2])
        R_2 = wp.mat33(R_1[0,0], R_1[0,1], R_1[0,2], 0.0, -R_1[1,1], -R_1[1,2], 0.0, 0.0, -R_1[2,2])
    else:
        Q = Q_1
        R_2 = R_1
    if R_2[2,2] > 1.0:
        R = wp.mat33(R_2[0,0], R_2[0,1], R_2[0,2], R_2[1,0], R_2[1,1], R_2[1,2], 0.0, 0.0, 1.0)
    else:
        fn = model.kappa[p] * (1.0 - R_2[2,2]) * (1.0 - R_2[2,2])
        ff = model.gamma[p] * wp.sqrt(R_2[0,2] * R_2[0,2] + R_2[1,2] * R_2[1,2])
        if ff > model.friction_coeff * fn:
            R = wp.mat33(R_2[0,0], R_2[0,1], R_2[0,2] * model.friction_coeff * fn / ff, R_2[1,0], R_2[1,1], R_2[1,2] * model.friction_coeff * fn / ff, R_2[2,0], R_2[2,1], R_2[2,2])
        else:
            R = R_2
    
    d3 = Q * wp.vec3(R[0,2], R[1,2], R[2,2])
    new_d = wp.mat33(d[0,0], d[0,1], d3[0], d[1,0], d[1,1], d3[1], d[2,0], d[2,1], d3[2])

    return new_d


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[p] = model.yield_stress[p] - model.softening * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[p] <= 0:
            model.mu[p] = 0.0
            model.lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


# for toothpaste
@wp.func
def viscoplasticity_return_mapping_with_StVK(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, dt: float
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    b_trial = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = wp.length(s_trial)
    y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[p]
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity / (2.0 * mu_hat * dt)
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[p]) * s_new + wp.vec3(
            trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
        )
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic
    else:
        return F_trial


@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic


@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    model.mu[p] = model.E[p] / (2.0 * (1.0 + model.nu[p]))
    model.lam[p] = (
        model.E[p] * model.nu[p] / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))
    )


@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    # state.grid_v_final[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.func
def update_cov_differentiable(
    state: MPMStateStruct,
    next_state: MPMStateStruct,
    p: int,
    grad_v: wp.mat33,
    dt: float,
):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    next_state.particle_cov[p * 6] = cov_np1[0, 0]
    next_state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    next_state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    next_state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    next_state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    next_state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float, offset: int):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    # output:               grid_v_in, grid_m
    p = wp.tid()
    if state.particle_selection[p] == 0:
        if state.particle_vertices[p] == 1:
            vertex_force = state.vertex_force[p-offset]
        elif state.particle_traditional[p] == 1:
            stress = state.particle_vol[p] * state.particle_stress[p]
        else:
            stress = state.particle_stress[p]
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    dpos = (
                        wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    ) * model.dx
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    dweight = compute_dweight(model, w, dw, i, j, k)

                    C = state.particle_C[p]
                    # if model.rpic = 0, standard apic
                    C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                        C - wp.transpose(C)
                    )

                    # C = (1.0 - model.rpic_damping) * state.particle_C[
                    #     p
                    # ] + model.rpic_damping / 2.0 * (
                    #     state.particle_C[p] - wp.transpose(state.particle_C[p])
                    # )

                    if model.rpic_damping < -0.001:
                        # standard pic
                        C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    if state.particle_vertices[p] == 1:
                        force = weight * vertex_force
                    else:
                        force = -stress * dweight
                    v_in_add = (
                        weight
                        * state.particle_mass[p]
                        * (state.particle_v[p] + C * dpos)
                        + dt * force
                    )
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )


# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (
            1.0 / state.grid_m[grid_x, grid_y, grid_z]
        )
        # add gravity
        v_out = v_out + dt * model.gravitational_accelaration
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


# @wp.kernel
# def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
#     p = wp.tid()
#     if state.particle_selection[p] == 0:
#         grid_pos = state.particle_x[p] * model.inv_dx
#         base_pos_x = wp.int(grid_pos[0] - 0.5)
#         base_pos_y = wp.int(grid_pos[1] - 0.5)
#         base_pos_z = wp.int(grid_pos[2] - 0.5)
#         fx = grid_pos - wp.vec3(
#             wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
#         )
#         wa = wp.vec3(1.5) - fx
#         wb = fx - wp.vec3(1.0)
#         wc = fx - wp.vec3(0.5)
#         w = wp.mat33(
#             wp.cw_mul(wa, wa) * 0.5,
#             wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
#             wp.cw_mul(wc, wc) * 0.5,
#         )
#         dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
#         new_v = wp.vec3(0.0, 0.0, 0.0)
#         new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#         new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

#         for i in range(0, 3):
#             for j in range(0, 3):
#                 for k in range(0, 3):
#                     ix = base_pos_x + i
#                     iy = base_pos_y + j
#                     iz = base_pos_z + k
#                     dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
#                     weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
#                     grid_v = state.grid_v_out[ix, iy, iz]
#                     new_v = new_v + grid_v * weight
#                     new_C = new_C + wp.outer(grid_v, dpos) * (
#                         weight * model.inv_dx * 4.0
#                     )
#                     dweight = compute_dweight(model, w, dw, i, j, k)
#                     new_F = new_F + wp.outer(grid_v, dweight)

#         state.particle_v[p] = new_v
#         # state.particle_x[p] = state.particle_x[p] + dt * new_v
#         # state.particle_x[p] = state.particle_x[p] + dt * state.particle_v[p]

#         # wp.atomic_add(state.particle_x, p, dt * state.particle_v[p]) # old one is this..
#         wp.atomic_add(state.particle_x, p, dt * new_v)  # debug
#         # new_x = state.particle_x[p] + dt * state.particle_v[p]
#         # state.particle_x[p] = new_x

#         state.particle_C[p] = new_C

#         I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
#         F_tmp = (I33 + new_F * dt) * state.particle_F[p]
#         state.particle_F_trial[p] = F_tmp
#         # debug for jelly
#         # wp.atomic_add(state.particle_F_trial, p, new_F * dt * state.particle_F[p])

#         if model.update_cov_with_F:
#             update_cov(state, p, new_F, dt)


# @wp.kernel
# def g2p_differentiable(
#     state: MPMStateStruct, next_state: MPMStateStruct, model: MPMModelStruct, dt: float
# ):
#     """
#     Compute:
#         next_state.particle_v, next_state.particle_x, next_state.particle_C, next_state.particle_F_trial
#     """
#     p = wp.tid()
#     if state.particle_selection[p] == 0:
#         grid_pos = state.particle_x[p] * model.inv_dx
#         base_pos_x = wp.int(grid_pos[0] - 0.5)
#         base_pos_y = wp.int(grid_pos[1] - 0.5)
#         base_pos_z = wp.int(grid_pos[2] - 0.5)
#         fx = grid_pos - wp.vec3(
#             wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
#         )
#         wa = wp.vec3(1.5) - fx
#         wb = fx - wp.vec3(1.0)
#         wc = fx - wp.vec3(0.5)
#         w = wp.mat33(
#             wp.cw_mul(wa, wa) * 0.5,
#             wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
#             wp.cw_mul(wc, wc) * 0.5,
#         )
#         dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
#         new_v = wp.vec3(0.0, 0.0, 0.0)
#         # new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#         new_C = wp.mat33(new_v, new_v, new_v)
        
#         new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

#         for i in range(0, 3):
#             for j in range(0, 3):
#                 for k in range(0, 3):
#                     ix = base_pos_x + i
#                     iy = base_pos_y + j
#                     iz = base_pos_z + k
#                     dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
#                     weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
#                     grid_v = state.grid_v_out[ix, iy, iz]
#                     new_v = (
#                         new_v + grid_v * weight
#                     )  # TODO, check gradient from static loop
#                     new_C = new_C + wp.outer(grid_v, dpos) * (
#                         weight * model.inv_dx * 4.0
#                     )
#                     dweight = compute_dweight(model, w, dw, i, j, k)
#                     new_F = new_F + wp.outer(grid_v, dweight)

#         next_state.particle_v[p] = new_v

#         # add clip here:
#         new_x = state.particle_x[p] + dt * new_v
#         dx = 1.0 / model.inv_dx
#         a_min = dx * 2.0
#         a_max = model.grid_lim - dx * 2.0

#         new_x_clamped = wp.vec3(
#             wp.clamp(new_x[0], a_min, a_max),
#             wp.clamp(new_x[1], a_min, a_max),
#             wp.clamp(new_x[2], a_min, a_max),
#         )
#         next_state.particle_x[p] = new_x_clamped

#         # next_state.particle_x[p] = new_x

#         next_state.particle_C[p] = new_C

#         I33_1 = wp.vec3(1.0, 0.0, 0.0)
#         I33_2 = wp.vec3(0.0, 1.0, 0.0)
#         I33_3 = wp.vec3(0.0, 0.0, 1.0)
#         I33 = wp.mat33(I33_1, I33_2, I33_3)
#         F_tmp = (I33 + new_F * dt) * state.particle_F[p]
#         next_state.particle_F_trial[p] = F_tmp

#         if 0:
#             update_cov_differentiable(state, next_state, p, new_F, dt)


@wp.kernel
def g2p_v(
    state: MPMStateStruct, model: MPMModelStruct, dt: float, offset: int,
):
    """
    Compute:
        next_state.particle_v, next_state.particle_x, next_state.particle_C
    """
    p = wp.tid()
    if state.particle_selection[p+offset] == 0:
        grid_pos = state.particle_x[p+offset] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        # new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_C = wp.mat33(new_v, new_v, new_v)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = (
                        new_v + grid_v * weight
                    )  # TODO, check gradient from static loop
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p+offset] = new_v

        # add clip here:
        new_x = state.particle_x[p+offset] + dt * new_v
        dx = 1.0 / model.inv_dx
        a_min = dx * 2.0
        a_max = model.grid_lim - dx * 2.0

        new_x_clamped = wp.vec3(
            wp.clamp(new_x[0], a_min, a_max),
            wp.clamp(new_x[1], a_min, a_max),
            wp.clamp(new_x[2], a_min, a_max),
        )
        state.particle_x[p+offset] = new_x_clamped
        # state.particle_x[p+offset] = new_x

        state.particle_C[p+offset] = new_C

        if state.particle_traditional[p+offset] == 1:
            I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            F_tmp = (I33 + new_F * dt) * state.particle_F[p+offset]
            state.particle_F_trial[p+offset] = F_tmp

@wp.kernel
def g2p_e(
    state: MPMStateStruct, model: MPMModelStruct, dt: float, offset: int,
):
    """
    Compute:
        next_state.particle_v, next_state.particle_x, next_state.particle_C, next_state.particle_d_trial
    """
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        # new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_C = wp.mat33(new_v, new_v, new_v)
        
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = (
                        new_v + grid_v * weight
                    )  # TODO, check gradient from static loop
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        face = state.faces[p]
        v1, v2, v3 = int(face[0])+offset, int(face[1])+offset, int(face[2])+offset

        state.particle_v[p] = (state.particle_v[v1] + state.particle_v[v2] + state.particle_v[v3]) / 3.0
        state.particle_x[p] = (state.particle_x[v1] + state.particle_x[v2] + state.particle_x[v3]) / 3.0
        state.particle_C[p] = new_C

        d1 = state.particle_x[v2] - state.particle_x[v1]
        d2 = state.particle_x[v3] - state.particle_x[v1]

        d = state.particle_d[p]
        d3 = wp.vec3(d[0,2], d[1,2], d[2,2])

        I33_1 = wp.vec3(1.0, 0.0, 0.0)
        I33_2 = wp.vec3(0.0, 1.0, 0.0)
        I33_3 = wp.vec3(0.0, 0.0, 1.0)
        I33 = wp.mat33(I33_1, I33_2, I33_3)
        d3_tmp = (I33 + new_F * dt) * d3
        new_d = wp.mat33(d1[0], d2[0], d3_tmp[0], d1[1], d2[1], d3_tmp[1], d1[2], d2[2], d3_tmp[2])
        state.particle_d[p] = new_d

@wp.kernel
def g2p_v_differentiable(
    state: MPMStateStruct, next_state: MPMSmallStateStruct, model: MPMModelStruct, dt: float, offset: int,
):
    """
    Compute:
        next_state.particle_v, next_state.particle_x, next_state.particle_C
    """
    p = wp.tid()
    if state.particle_selection[p+offset] == 0:
        grid_pos = state.particle_x[p+offset] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        new_v = wp.vec3(0.0, 0.0, 0.0)
        # new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_C = wp.mat33(new_v, new_v, new_v)

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = (
                        new_v + grid_v * weight
                    )  # TODO, check gradient from static loop
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )

        next_state.particle_v[p+offset] = new_v

        # add clip here:
        new_x = state.particle_x[p+offset] + dt * new_v
        dx = 1.0 / model.inv_dx
        a_min = dx * 2.0
        a_max = model.grid_lim - dx * 2.0

        new_x_clamped = wp.vec3(
            wp.clamp(new_x[0], a_min, a_max),
            wp.clamp(new_x[1], a_min, a_max),
            wp.clamp(new_x[2], a_min, a_max),
        )
        next_state.particle_x[p+offset] = new_x_clamped

        # next_state.particle_x[p] = new_x

        next_state.particle_C[p+offset] = new_C


@wp.kernel
def g2p_e_differentiable(
    state: MPMStateStruct, next_state: MPMSmallStateStruct, model: MPMModelStruct, dt: float, offset: int,
):
    """
    Compute:
        next_state.particle_v, next_state.particle_x, next_state.particle_C, next_state.particle_d_trial
    """
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        # new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_C = wp.mat33(new_v, new_v, new_v)
        
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = (
                        new_v + grid_v * weight
                    )  # TODO, check gradient from static loop
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        face = state.faces[p]
        v1, v2, v3 = int(face[0])+offset, int(face[1])+offset, int(face[2])+offset

        next_state.particle_v[p] = (next_state.particle_v[v1] + next_state.particle_v[v2] + next_state.particle_v[v3]) / 3.0
        next_state.particle_x[p] = (next_state.particle_x[v1] + next_state.particle_x[v2] + next_state.particle_x[v3]) / 3.0
        next_state.particle_C[p] = new_C

        d1 = next_state.particle_x[v2] - next_state.particle_x[v1]
        d2 = next_state.particle_x[v3] - next_state.particle_x[v1]

        d = state.particle_d[p]
        d3 = wp.vec3(d[0,2], d[1,2], d[2,2])

        I33_1 = wp.vec3(1.0, 0.0, 0.0)
        I33_2 = wp.vec3(0.0, 1.0, 0.0)
        I33_3 = wp.vec3(0.0, 0.0, 1.0)
        I33 = wp.mat33(I33_1, I33_2, I33_3)
        d3_tmp = (I33 + new_F * dt) * d3
        new_d = wp.mat33(d1[0], d2[0], d3_tmp[0], d1[1], d2[1], d3_tmp[1], d1[2], d2[2], d3_tmp[2])
        next_state.particle_d[p] = new_d


@wp.kernel
def clip_particle_x(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    posx = state.particle_x[p]
    if state.particle_selection[p] == 0:
        dx = 1.0 / model.inv_dx
        a_min = dx * 2.0
        a_max = model.grid_lim - dx * 2.0
        new_x = wp.vec3(
            wp.clamp(posx[0], a_min, a_max),
            wp.clamp(posx[1], a_min, a_max),
            wp.clamp(posx[2], a_min, a_max),
        )

        state.particle_x[
            p
        ] = new_x  # Warn: this gives wrong gradient, don't use this for backward


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    """
    state.particle_F_trial => state.particle_F   # return mapping
    state.particle_F => state.particle_stress    # stress-strain

    TODO: check the gradient of SVD!  is wp.svd3 differentiable? I guess so
    """
    p = wp.tid()
    if state.particle_selection[p] == 0:
        # apply return mapping
        if state.particle_elements[p] == 1:
            state.particle_d[p] = anisotropy_return_mapping(
                state.particle_d[p], model, p
            )

            stress = kirchoff_stress_Anisotropy(
                state.particle_R_inv[p],
                state.particle_d[p],
                state.faces[p],
                state.particle_vol[p],
                state,
                model.mu[p],
                model.lam[p],
                model.gamma[p],
                model.kappa[p]
            )
        
        elif state.particle_traditional[p] == 1:

            # state.particle_F[p] = sand_return_mapping(
            #     state.particle_F_trial[p], state, model, p
            # )
            # state.particle_F[p] = state.particle_F_trial[p]
            if model.material == 1:  # metal
                state.particle_F[p] = von_mises_return_mapping(
                    state.particle_F_trial[p], model, p
                )
            elif model.material == 2:  # sand
                state.particle_F[p] = sand_return_mapping(
                    state.particle_F_trial[p], state, model, p
                )
            elif model.material == 3:  # visplas, with StVk+VM, no thickening
                state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
                    state.particle_F_trial[p], model, p, dt
                )
            elif model.material == 5:
                state.particle_F[p] = von_mises_return_mapping_with_damage(
                    state.particle_F_trial[p], model, p
                )
            else:  # elastic
                state.particle_F[p] = state.particle_F_trial[p]

            J = wp.determinant(state.particle_F[p])
            U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            sig = wp.vec3(0.0)
            stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            wp.svd3(state.particle_F[p], U, sig, V)

            if model.material == 0 or model.material == 5:
                stress = kirchoff_stress_FCR(
                    state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
                )
            if model.material == 1:
                stress = kirchoff_stress_StVK(
                    state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
                )
            if model.material == 2:
                stress = kirchoff_stress_drucker_prager(
                    state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
                )
            if model.material == 3:
                # temporarily use stvk, subject to change
                stress = kirchoff_stress_StVK(
                    state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
                )
            
            # stress = kirchoff_stress_drucker_prager(
            #     state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            # )
            # stress = kirchoff_stress_FCR(
            #     state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            # )
            stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry

        state.particle_stress[p] = stress


@wp.kernel
def compute_cov_from_F(state: MPMStateStruct, new_cov: wp.array(dtype=float)):
    p = wp.tid()

    F = state.particle_F_trial[p]

    init_cov = wp.mat33(0.0)
    init_cov[0, 0] = state.particle_cov[p * 6]
    init_cov[0, 1] = state.particle_cov[p * 6 + 1]
    init_cov[0, 2] = state.particle_cov[p * 6 + 2]
    init_cov[1, 0] = state.particle_cov[p * 6 + 1]
    init_cov[1, 1] = state.particle_cov[p * 6 + 3]
    init_cov[1, 2] = state.particle_cov[p * 6 + 4]
    init_cov[2, 0] = state.particle_cov[p * 6 + 2]
    init_cov[2, 1] = state.particle_cov[p * 6 + 4]
    init_cov[2, 2] = state.particle_cov[p * 6 + 5]

    cov = F * init_cov * wp.transpose(F)

    new_cov[p * 6] = cov[0, 0]
    new_cov[p * 6 + 1] = cov[0, 1]
    new_cov[p * 6 + 2] = cov[0, 2]
    new_cov[p * 6 + 3] = cov[1, 1]
    new_cov[p * 6 + 4] = cov[1, 2]
    new_cov[p * 6 + 5] = cov[2, 2]


# @wp.kernel
# def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
#     p = wp.tid()

#     F = state.particle_F_trial[p]

#     # polar svd decomposition
#     U = wp.mat33(0.0)
#     V = wp.mat33(0.0)
#     sig = wp.vec3(0.0)
#     wp.svd3(F, U, sig, V)

#     if wp.determinant(U) < 0.0:
#         U[0, 2] = -U[0, 2]
#         U[1, 2] = -U[1, 2]
#         U[2, 2] = -U[2, 2]

#     if wp.determinant(V) < 0.0:
#         V[0, 2] = -V[0, 2]
#         V[1, 2] = -V[1, 2]
#         V[2, 2] = -V[2, 2]

#     # compute rotation matrix
#     R = U * wp.transpose(V)
#     state.particle_R[p] = wp.transpose(R) # particle R is removed


@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    grid_x, grid_y, grid_z = wp.tid()
    # state.grid_v_out[grid_x, grid_y, grid_z] = (
    #     state.grid_v_out[grid_x, grid_y, grid_z] * scale
    # )
    wp.atomic_sub(
        state.grid_v_out,
        grid_x,
        grid_y,
        grid_z,
        (1.0 - scale) * state.grid_v_out[grid_x, grid_y, grid_z],
    )


@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        model.E[p] = params_modifier.E
        model.nu[p] = params_modifier.nu
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(
    state: MPMStateStruct, impulse_modifier: Impulse_modifier
):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
    ):
        impulse_modifier.mask[p] = 1
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_cylinder(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(
        offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal
    )
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def compute_position_l2_loss(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    pos = mpm_state.particle_x[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    l2 = wp.length(pos - pos_gt)

    wp.atomic_add(loss, 0, l2)


@wp.kernel
def aggregate_grad(x: wp.array(dtype=float), grad: wp.array(dtype=float)):
    tid = wp.tid()

    # gradient descent step
    wp.atomic_add(x, 0, grad[tid])


@wp.kernel
def set_F_C_p2g(
    state: MPMStateStruct, model: MPMModelStruct, target_pos: wp.array(dtype=wp.vec3)
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        # p2g for displacement
        particle_disp = target_pos[p] - state.particle_x[p]
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    v_in_add = weight * state.particle_mass[p] * particle_disp
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )


# @wp.kernel
# def set_F_C_g2p(state: MPMStateStruct, model: MPMModelStruct):
#     p = wp.tid()
#     if state.particle_selection[p] == 0:
#         grid_pos = state.particle_x[p] * model.inv_dx
#         base_pos_x = wp.int(grid_pos[0] - 0.5)
#         base_pos_y = wp.int(grid_pos[1] - 0.5)
#         base_pos_z = wp.int(grid_pos[2] - 0.5)
#         fx = grid_pos - wp.vec3(
#             wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
#         )
#         wa = wp.vec3(1.5) - fx
#         wb = fx - wp.vec3(1.0)
#         wc = fx - wp.vec3(0.5)
#         w = wp.mat33(
#             wp.cw_mul(wa, wa) * 0.5,
#             wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
#             wp.cw_mul(wc, wc) * 0.5,
#         )
#         dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
#         new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#         new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

#         # g2p for C and F
#         for i in range(0, 3):
#             for j in range(0, 3):
#                 for k in range(0, 3):
#                     ix = base_pos_x + i
#                     iy = base_pos_y + j
#                     iz = base_pos_z + k
#                     dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
#                     weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
#                     grid_v = state.grid_v_out[ix, iy, iz]
#                     new_C = new_C + wp.outer(grid_v, dpos) * (
#                         weight * model.inv_dx * 4.0
#                     )
#                     dweight = compute_dweight(model, w, dw, i, j, k)
#                     new_F = new_F + wp.outer(grid_v, dweight)

#         # C should still be zero..
#         # state.particle_C[p] = new_C
#         I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
#         F_tmp = I33 + new_F
#         state.particle_F_trial[p] = F_tmp

#         if model.update_cov_with_F:
#             update_cov(state, p, new_F, 1.0)


@wp.kernel
def compute_posloss_with_grad(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3),
    grad: wp.array(dtype=wp.vec3),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    pos = mpm_state.particle_x[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    # l2 = wp.length(pos - (pos_gt - grad[tid] * dt))
    diff = pos - (pos_gt - grad[tid] * dt)
    l2 = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, l2)


@wp.kernel
def compute_veloloss_with_grad(
    mpm_state: MPMStateStruct,
    gt_pos: wp.array(dtype=wp.vec3),
    grad: wp.array(dtype=wp.vec3),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    pos = mpm_state.particle_v[tid]
    pos_gt = gt_pos[tid]

    # l1_diff = wp.abs(pos - pos_gt)
    # l2 = wp.length(pos - (pos_gt - grad[tid] * dt))

    diff = pos - (pos_gt - grad[tid] * dt)
    l2 = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, l2)

# @wp.kernel
# def compute_Rinvloss_with_grad(
#     mpm_state: MPMStateStruct,
#     gt_pos: wp.array(dtype=wp.vec3),
#     grad: wp.array(dtype=wp.vec3),
#     dt: float,
#     loss: wp.array(dtype=float),
# ):
#     tid = wp.tid()

#     pos = mpm_state.particle_R_inv[tid]
#     pos_gt = gt_pos[tid]

#     # l1_diff = wp.abs(pos - pos_gt)
#     # l2 = wp.length(pos - (pos_gt - grad[tid] * dt))

#     diff = pos - (pos_gt - grad[tid] * dt)
#     l2 = wp.dot(diff, diff)
#     wp.atomic_add(loss, 0, l2)

@wp.kernel
def compute_dloss_with_grad(
    mpm_state: MPMStateStruct,
    gt_mat: wp.array(dtype=wp.mat33),
    grad: wp.array(dtype=wp.mat33),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    mat_ = mpm_state.particle_d[tid]
    mat_gt = gt_mat[tid]

    mat_gt = mat_gt - grad[tid] * dt
    # l1_diff = wp.abs(pos - pos_gt)
    mat_diff = mat_ - mat_gt

    l2 = wp.ddot(mat_diff, mat_diff)
    # l2 = wp.sqrt(
    #     mat_diff[0, 0] ** 2.0
    #     + mat_diff[0, 1] ** 2.0
    #     + mat_diff[0, 2] ** 2.0
    #     + mat_diff[1, 0] ** 2.0
    #     + mat_diff[1, 1] ** 2.0
    #     + mat_diff[1, 2] ** 2.0
    #     + mat_diff[2, 0] ** 2.0
    #     + mat_diff[2, 1] ** 2.0
    #     + mat_diff[2, 2] ** 2.0
    # )

    wp.atomic_add(loss, 0, l2)


@wp.kernel
def compute_dloss_with_grad_3(
    mpm_state: MPMStateStruct,
    gt_mat: wp.array(dtype=wp.mat33),
    grad: wp.array(dtype=wp.mat33),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    mat_ = mpm_state.particle_d[tid]
    mat_3 = wp.vec3(mat_[0,2], mat_[1,2], mat_[2,2])

    mat_gt_ = gt_mat[tid]
    mat_gt_3 = wp.vec3(mat_gt_[0,2], mat_gt_[1,2], mat_gt_[2,2])

    grad_ = grad[tid]
    grad_3 = wp.vec3(grad_[0,2], grad_[1,2], grad_[2,2])

    mat_gt = mat_gt_3 - grad_3 * dt
    # l1_diff = wp.abs(pos - pos_gt)
    mat_diff = mat_3 - mat_gt

    l2 = wp.dot(mat_diff, mat_diff)
    # l2 = wp.sqrt(
    #     mat_diff[0, 0] ** 2.0
    #     + mat_diff[0, 1] ** 2.0
    #     + mat_diff[0, 2] ** 2.0
    #     + mat_diff[1, 0] ** 2.0
    #     + mat_diff[1, 1] ** 2.0
    #     + mat_diff[1, 2] ** 2.0
    #     + mat_diff[2, 0] ** 2.0
    #     + mat_diff[2, 1] ** 2.0
    #     + mat_diff[2, 2] ** 2.0
    # )

    wp.atomic_add(loss, 0, l2)


@wp.kernel
def compute_Closs_with_grad(
    mpm_state: MPMStateStruct,
    gt_mat: wp.array(dtype=wp.mat33),
    grad: wp.array(dtype=wp.mat33),
    dt: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    mat_ = mpm_state.particle_C[tid]
    mat_gt = gt_mat[tid]

    mat_gt = mat_gt - grad[tid] * dt
    # l1_diff = wp.abs(pos - pos_gt)

    mat_diff = mat_ - mat_gt
    l2 = wp.ddot(mat_diff, mat_diff)

    wp.atomic_add(loss, 0, l2)

@wp.kernel()
def sum_float(x: wp.array(dtype=float), y: wp.array(dtype=float), z: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(z, 0, x[tid]*y[tid])

@wp.kernel()
def sum_vec3(x: wp.array(dtype=wp.vec3), y: wp.array(dtype=wp.vec3), z: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(z, 0, x[tid][0]*y[tid][0] + x[tid][1]*y[tid][1] + x[tid][2]*y[tid][2])

@wp.kernel()
def sum_mat33(x: wp.array(dtype=wp.mat33), y: wp.array(dtype=wp.mat33), z: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(z, 0, x[tid][0,0]*y[tid][0,0] + x[tid][0,1]*y[tid][0,1] + x[tid][0,2]*y[tid][0,2] + x[tid][1,0]*y[tid][1,0] + x[tid][1,1]*y[tid][1,1] + x[tid][1,2]*y[tid][1,2] + x[tid][2,0]*y[tid][2,0] + x[tid][2,1]*y[tid][2,1] + x[tid][2,2]*y[tid][2,2])