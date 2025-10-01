import sys
import os

import warp as wp

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from mpm_data_structure import *
from mpm_utils import *
from typing import Optional, Union, Sequence, Any, Tuple
from jaxtyping import Float, Int, Shaped


class MPMWARP(object):
    def __init__(self, n_particles, n_elements, n_vertices, n_grid=100, grid_lim=1.0, mesh_vertices=None, mesh_faces=None, num_joint_t=0, num_joint_v=0, num_joint_f=0, device="cuda:0"):
        self.initialize(n_particles, n_elements, n_vertices, n_grid, grid_lim, mesh_vertices, mesh_faces, num_joint_t, num_joint_v, num_joint_f, device=device)
        self.time_profile = {}

    def initialize(self, n_particles, n_elements, n_vertices, n_grid=100, grid_lim=1.0, mesh_vertices=None, mesh_faces=None, num_joint_t=0, num_joint_v=0, num_joint_f=0, device="cuda:0"):
        self.n_particles = n_particles
        self.n_elements = n_elements
        self.n_vertices = n_vertices
        n_no_vertices = n_particles - n_vertices
        self.n_no_vertices = n_no_vertices
        self.num_joint_t = num_joint_t
        self.num_joint_v = num_joint_v
        self.num_joint_f = num_joint_f

        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []
        self.mesh_colliders = []
        self.mesh_collider_params = []
        self.particle_movers = []
        self.particle_mover_params = []

        self.tailored_struct_for_bc = MPMtailoredStruct()
        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []

        if mesh_vertices is not None and mesh_faces is not None:
            points = wp.from_numpy(mesh_vertices, dtype=wp.vec3, device=device, requires_grad=False)
            velocities = wp.zeros_like(points, requires_grad=False)
            indices = wp.from_numpy(mesh_faces.flatten(), dtype=wp.int32, device=device, requires_grad=False)
            self.mesh = wp.Mesh(points=points, velocities=velocities, indices=indices)
            self.num_mesh_v = mesh_vertices.shape[0]
            self.num_mesh_f = mesh_faces.shape[0]

    # must give density. mass will be updated as density * volume
    def set_parameters(self, device="cuda:0", **kwargs):
        self.set_parameters_dict(device, kwargs)

    def set_parameters_dict(self, mpm_model, mpm_state, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            if kwargs["material"] == "jelly":
                mpm_model.material = 0
            elif kwargs["material"] == "metal":
                mpm_model.material = 1
            elif kwargs["material"] == "sand":
                mpm_model.material = 2
            elif kwargs["material"] == "foam":
                mpm_model.material = 3
            elif kwargs["material"] == "snow":
                mpm_model.material = 4
            elif kwargs["material"] == "plasticine":
                mpm_model.material = 5
            elif kwargs["material"] == "neo-hookean":
                mpm_model.material = 6
            elif kwargs["material"] == "cloth":
                mpm_model.material = 7
            else:
                raise TypeError("Undefined material type")

        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_model.yield_stress, val],
                device=device,
            )
        if "hardening" in kwargs:
            mpm_model.hardening = kwargs["hardening"]
        if "xi" in kwargs:
            mpm_model.xi = kwargs["xi"]
        if "friction_angle" in kwargs:
            mpm_model.friction_angle = kwargs["friction_angle"]
            sin_phi = wp.sin(mpm_model.friction_angle / 180.0 * 3.14159265)
            mpm_model.friction_coeff = wp.tan(mpm_model.friction_angle / 180.0 * 3.14159265)
            mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        if "g" in kwargs:
            mpm_model.gravitational_accelaration = wp.vec3(
                kwargs["g"][0], kwargs["g"][1], kwargs["g"][2]
            )

        if "density" in kwargs:
            density_value = kwargs["density"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_state.particle_density, density_value],
                device=device,
            )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    mpm_state.particle_density,
                    mpm_state.particle_vol,
                    mpm_state.particle_mass,
                ],
                device=device,
            )
        if "rpic_damping" in kwargs:
            mpm_model.rpic_damping = kwargs["rpic_damping"]
        if "plastic_viscosity" in kwargs:
            mpm_model.plastic_viscosity = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            mpm_model.softening = kwargs["softening"]
        if "grid_v_damping_scale" in kwargs:
            mpm_model.grid_v_damping_scale = kwargs["grid_v_damping_scale"]

    def set_E_nu(self, mpm_model, E: float, nu: float, gamma: float, kappa: float, device="cuda:0"):
        if isinstance(E, float):
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_model.E, E],
                device=device,
            )
        else:  # E is warp array
            wp.launch(
                kernel=set_float_vec_to_vec,
                dim=self.n_particles,
                inputs=[mpm_model.E, E],
                device=device,
            )

        if isinstance(nu, float):
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_model.nu, nu],
                device=device,
            )
        else:
            wp.launch(
                kernel=set_float_vec_to_vec,
                dim=self.n_particles,
                inputs=[mpm_model.nu, nu],
                device=device,
            )

        if isinstance(gamma, float):
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_model.gamma, gamma],
                device=device,
            )
        else:
            wp.launch(
                kernel=set_float_vec_to_vec,
                dim=self.n_particles,
                inputs=[mpm_model.gamma, gamma],
                device=device,
            )

        if isinstance(kappa, float):
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[mpm_model.kappa, kappa],
                device=device,
            )
        else:
            wp.launch(
                kernel=set_float_vec_to_vec,
                dim=self.n_particles,
                inputs=[mpm_model.kappa, kappa],
                device=device,
            )

    def set_E_nu_from_torch(
        self,
        mpm_model,
        E: Float[Tensor, "n"] | Float[Tensor, "1"],
        nu: Float[Tensor, "n"] | Float[Tensor, "1"],
        gamma: Float[Tensor, "n"] | Float[Tensor, "1"],
        kappa: Float[Tensor, "n"] | Float[Tensor, "1"],
        device="cuda:0",
    ):
        if E.ndim == 0:
            E_inp = E.item()  # float
        else:
            E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=True)

        if nu.ndim == 0:
            nu_inp = nu.item()  # float
        else:
            nu_inp = from_torch_safe(nu, dtype=wp.float32, requires_grad=True)

        if gamma.ndim == 0:
            gamma_inp = gamma.item()  # float
        else:
            gamma_inp = from_torch_safe(gamma, dtype=wp.float32, requires_grad=True)

        if kappa.ndim == 0:
            kappa_inp = kappa.item()  # float
        else:
            kappa_inp = from_torch_safe(kappa, dtype=wp.float32, requires_grad=True)

        self.set_E_nu(mpm_model, E_inp, nu_inp, gamma_inp, kappa_inp, device=device)

    def prepare_mu_lam(self, mpm_model, mpm_state, device="cuda:0"):
        # compute mu and lam from E and nu
        wp.launch(
            kernel=compute_mu_lam_from_E_nu,
            dim=self.n_particles,
            inputs=[mpm_state, mpm_model],
            device=device,
        )

    def p2g2p(
        self, mpm_model, mpm_state, dt, mesh_x=None, mesh_v=None, joint_traditional_v=None, joint_verts_v=None, joint_faces_v=None, device="cuda:0"
    ):
        """
        Some boundary conditions, might not give gradient,
        see kernels in
            self.pre_p2g_operations,    Usually None.
            self.particle_velocity_modifiers.   Mostly used to freeze points
            self.grid_postprocess,      Should apply BC here
        """
        grid_size = (
            mpm_model.grid_dim_x,
            mpm_model.grid_dim_y,
            mpm_model.grid_dim_z,
        )
        wp.launch(
            kernel=zero_grid,  # gradient might gone
            dim=(grid_size),
            inputs=[mpm_state, mpm_model],
            device=device,
        )

        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_vertices,
            inputs=[mpm_state.vertex_force],
            device=device,
        )

        # apply pre-p2g operations on particles
        # apply impulse force on particles..
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, mpm_state, self.impulse_params[k]],
                device=device,
            )

        # apply dirichlet particle v modifier
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    mpm_state,
                    self.particle_velocity_modifier_params[k],
                ],
                device=device,
            )
        
        if joint_traditional_v is not None:
            new_joint_traditional_v = wp.from_numpy(joint_traditional_v.detach().cpu().numpy(), dtype=wp.vec3, device=device)
            joint_num = joint_traditional_v.shape[0]

        if mesh_x is not None:
            new_mesh_points = wp.from_numpy(mesh_x.detach().cpu().numpy(), dtype=wp.vec3, device=device)

            with wp.ScopedTimer(
                "update_mesh_positions",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=set_vec3_to_vec3,
                    dim=self.num_mesh_v,
                    inputs=[self.mesh.points, new_mesh_points],
                    device=device,
                )

        if mesh_v is not None:
            new_mesh_velocities = wp.from_numpy(mesh_v.detach().cpu().numpy(), dtype=wp.vec3, device=device)

            with wp.ScopedTimer(
                "update_mesh_velocities",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=set_vec3_to_vec3,
                    dim=self.num_mesh_v,
                    inputs=[self.mesh.velocities, new_mesh_velocities],
                    device=device,
                )

        # compute stress = stress(returnMap(F_trial))
        # F_trail => F                    # TODO: this is overite..
        # F, SVD(F), lam, mu => Stress.   # TODO: this is overite..

        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_no_vertices,
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )  # F and stress are updated
        
        # print("="*40)
        # print("TIME:", self.time)
        # print("="*40)
        # print("particle_d_trial:", np.abs(mpm_state.particle_d_trial.numpy()).max())
        # print("particle_d      :", np.abs(mpm_state.particle_d.numpy()).max())
        # print("particle_d_diff :", np.abs(mpm_state.particle_d.numpy()-mpm_state.particle_d_trial.numpy()).max())
        # print("particle_stress :", np.abs(mpm_state.particle_stress.numpy()).max())
        # print("vertex_force    :", np.abs(mpm_state.vertex_force.numpy()).max())
        # print("="*40)
        
        # for k, v in vars(mpm_state).items():
        #     if k.startswith("particle") or k.startswith("vertex") or k.startswith("grid"):
        #         print(k, v.numpy().shape)

        # p2g
        with wp.ScopedTimer(
            "p2g",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[mpm_state, mpm_model, dt, self.n_no_vertices],
                device=device,
            )  # apply p2g'

        # grid update
        with wp.ScopedTimer(
            "grid_update", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )

        if mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=(grid_size),
                inputs=[mpm_state, mpm_model.grid_v_damping_scale],
                device=device,
            )

        # apply Mesh Collision on grid
        with wp.ScopedTimer(
            "apply_Mesh_Collision_on_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            for k in range(len(self.mesh_colliders)):
                wp.launch(
                    kernel=self.mesh_colliders[k][0],
                    dim=grid_size,
                    inputs=[
                        self.mesh_collider_params[k],
                    ],
                    device=device,
                )
                wp.launch(
                    kernel=self.mesh_colliders[k][1],
                    dim=self.num_mesh_f,
                    inputs=[
                        self.mesh_collider_params[k],
                        mpm_model,
                    ],
                    device=device,
                )
                wp.launch(
                    kernel=self.mesh_colliders[k][2],
                    dim=grid_size,
                    inputs=[
                        self.mesh_collider_params[k],
                    ],
                    device=device,
                )
                wp.launch(
                    kernel=self.mesh_colliders[k][3],
                    dim=grid_size,
                    inputs=[
                        mpm_state,
                        self.mesh_collider_params[k],
                    ],
                    device=device,
                )
        
        if joint_verts_v is not None and joint_faces_v is not None:
            new_joint_verts_v = wp.from_numpy(joint_verts_v.detach().cpu().numpy(), dtype=wp.vec3, device=device)
            new_joint_faces_v = wp.from_numpy(joint_faces_v.detach().cpu().numpy(), dtype=wp.vec3, device=device)
            # apply Particle Moving on grid
            with wp.ScopedTimer(
                "apply_Particle_Moving_on_grid", synchronize=True, print=False, dict=self.time_profile
            ):
                for k in range(len(self.particle_movers)):
                    wp.launch(
                        kernel=self.particle_movers[k][0],
                        dim=grid_size,
                        inputs=[
                            self.particle_mover_params[k],
                        ],
                        device=device,
                    )
                    if joint_traditional_v is not None:
                        wp.launch(
                            kernel=self.particle_movers[k][1],
                            dim=joint_num,
                            inputs=[
                                new_joint_traditional_v,
                                self.particle_mover_params[k],
                                mpm_state,
                                mpm_model,
                                self.n_particles-self.n_vertices-joint_num,
                            ],
                            device=device,
                        )
                    wp.launch(
                        kernel=self.particle_movers[k][2],
                        dim=self.num_joint_v,
                        inputs=[
                            new_joint_verts_v,
                            self.particle_mover_params[k],
                            mpm_state,
                            mpm_model,
                            self.n_no_vertices,
                        ],
                        device=device,
                    )
                    wp.launch(
                        kernel=self.particle_movers[k][3],
                        dim=self.num_joint_f,
                        inputs=[
                            new_joint_faces_v,
                            self.particle_mover_params[k],
                            mpm_state,
                            mpm_model,
                        ],
                        device=device,
                    )
                    wp.launch(
                        kernel=self.particle_movers[k][4],
                        dim=grid_size,
                        inputs=[
                            mpm_state,
                            self.particle_mover_params[k],
                        ],
                        device=device,
                    )

        # apply BC on grid, collide
        with wp.ScopedTimer(
            "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        mpm_state,
                        mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])

        # # g2p
        # with wp.ScopedTimer(
        #     "g2p", synchronize=True, print=False, dict=self.time_profile
        # ):
        #     wp.launch(
        #         kernel=g2p_differentiable,
        #         dim=self.n_particles,
        #         inputs=[mpm_state, next_state, mpm_model, dt],
        #         device=device,
        #     )  # x, v, C, F_trial are updated

        # g2p_v
        with wp.ScopedTimer(
            "g2p_v", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p_v,
                dim=(self.n_particles-self.n_elements),
                inputs=[mpm_state, mpm_model, dt, self.n_elements],
                device=device,
            )  # x, v, C, F_trial are updated

        # g2p_e
        with wp.ScopedTimer(
            "g2p_e", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p_e,
                dim=self.n_elements,
                inputs=[mpm_state, mpm_model, dt, self.n_no_vertices],
                device=device,
            )  # x, v, C, F_trial are updated

        self.time = self.time + dt

    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))
    
    def export_particle_cov_to_torch(self, mpm_state, device="cuda:0"):
        new_cov = wp.zeros(
            shape=self.n_no_vertices * 6, dtype=float, device=device, requires_grad=False
        )
        with wp.ScopedTimer(
            "compute_cov_from_F",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_cov_from_F,
                dim=self.n_no_vertices,
                inputs=[mpm_state, new_cov],
                device=device,
            )

        cov = wp.to_torch(new_cov).clone()
        return cov

    # a surface specified by a point and the normal vector
    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "cut":
            collider_param.surface_type = 11
        else:
            collider_param.surface_type = 2
        # frictional
        collider_param.friction = friction

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    elif param.surface_type == 11:
                        if (
                            float(grid_z) * model.dx < 0.4
                            or float(grid_z) * model.dx > 0.53
                        ):
                            state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                                0.0, 0.0, 0.0
                            )
                        else:
                            v_in = state.grid_v_out[grid_x, grid_y, grid_z]
                            state.grid_v_out[grid_x, grid_y, grid_z] = (
                                wp.vec3(v_in[0], 0.0, v_in[2]) * 0.3
                            )
                    else:
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = (
                                v - normal_component * n
                            )  # Project out all normal component
                        else:
                            v = (
                                v - wp.min(normal_component, 0.0) * n
                            )  # Project out only inward normal component
                        if normal_component < 0.0 and wp.length(v) > 1e-20:
                            v = wp.max(
                                0.0, wp.length(v) + normal_component * param.friction
                            ) * wp.normalize(
                                v
                            )  # apply friction here
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # move joint particles
    def add_particle_mover(
        self,
        n_grid
    ):
        mover_param = Particle_mover()
        mover_param.weight = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=float, device="cuda:0")
        mover_param.velocity = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device="cuda:0")

        @wp.kernel
        def zero_grid(
            param: Particle_mover
        ):
            grid_x, grid_y, grid_z = wp.tid()
            param.weight[grid_x, grid_y, grid_z] = 0.0
            param.velocity[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        @wp.kernel
        def add_velocity_traditional(
            particle_velocity: wp.array(dtype=wp.vec3),
            param: Particle_mover,
            state: MPMStateStruct,
            model: MPMModelStruct,
            offset: int,
        ):
            p = wp.tid()

            grid_pos = state.particle_x[p+offset] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)

            if base_pos_x >= 0 and base_pos_x < model.grid_dim_x - 3 and base_pos_y >= 0 and base_pos_y < model.grid_dim_y - 3 and base_pos_z >= 0 and base_pos_z < model.grid_dim_z - 3:
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

                for i in range(0, 3):
                    for j in range(0, 3):
                        for k in range(0, 3):
                            ix = base_pos_x + i
                            iy = base_pos_y + j
                            iz = base_pos_z + k
                            weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                            wp.atomic_add(param.velocity, ix, iy, iz, weight * particle_velocity[p])
                            wp.atomic_add(param.weight, ix, iy, iz, weight)

        @wp.kernel
        def add_velocity_verts(
            particle_velocity: wp.array(dtype=wp.vec3),
            param: Particle_mover,
            state: MPMStateStruct,
            model: MPMModelStruct,
            offset: int,
        ):
            p = wp.tid()

            grid_pos = state.particle_x[p+offset] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)

            if base_pos_x >= 0 and base_pos_x < model.grid_dim_x - 3 and base_pos_y >= 0 and base_pos_y < model.grid_dim_y - 3 and base_pos_z >= 0 and base_pos_z < model.grid_dim_z - 3:
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

                for i in range(0, 3):
                    for j in range(0, 3):
                        for k in range(0, 3):
                            ix = base_pos_x + i
                            iy = base_pos_y + j
                            iz = base_pos_z + k
                            weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                            wp.atomic_add(param.velocity, ix, iy, iz, weight * particle_velocity[p])
                            wp.atomic_add(param.weight, ix, iy, iz, weight)
        
        @wp.kernel
        def add_velocity_faces(
            particle_velocity: wp.array(dtype=wp.vec3),
            param: Particle_mover,
            state: MPMStateStruct,
            model: MPMModelStruct,
        ):
            p = wp.tid()

            grid_pos = state.particle_x[p] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)

            if base_pos_x >= 0 and base_pos_x < model.grid_dim_x - 3 and base_pos_y >= 0 and base_pos_y < model.grid_dim_y - 3 and base_pos_z >= 0 and base_pos_z < model.grid_dim_z - 3:
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

                for i in range(0, 3):
                    for j in range(0, 3):
                        for k in range(0, 3):
                            ix = base_pos_x + i
                            iy = base_pos_y + j
                            iz = base_pos_z + k
                            weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                            wp.atomic_add(param.velocity, ix, iy, iz, weight * particle_velocity[p])
                            wp.atomic_add(param.weight, ix, iy, iz, weight)
        
        @wp.kernel
        def normalize_grid(
            state: MPMStateStruct,
            param: Particle_mover,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if param.weight[grid_x, grid_y, grid_z] > 1e-15:
                inv_w = 1.0 / param.weight[grid_x, grid_y, grid_z]
                v = param.velocity[grid_x, grid_y, grid_z] * inv_w
                state.grid_v_out[grid_x, grid_y, grid_z] = v

        self.particle_movers.append([zero_grid, add_velocity_traditional, add_velocity_verts, add_velocity_faces, normalize_grid])
        self.particle_mover_params.append(mover_param)

    # a mesh specified by a mesh_id
    def add_mesh_collider(
        self,
        mesh_id,
        n_grid,
        friction=0.0,
    ):
        collider_param = Mesh_collider()
        collider_param.mesh_id = mesh_id
        collider_param.friction = friction
        collider_param.weight = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=float, device="cuda:0")
        collider_param.mesh_v_in = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device="cuda:0")
        collider_param.mesh_v_out = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device="cuda:0")
        collider_param.mesh_normal = wp.zeros(shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device="cuda:0")

        @wp.kernel
        def zero_grid(
            param: Mesh_collider
        ):
            grid_x, grid_y, grid_z = wp.tid()
            param.weight[grid_x, grid_y, grid_z] = 0.0
            param.mesh_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
            param.mesh_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
            param.mesh_normal[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        @wp.kernel
        def compute_mesh(
            param: Mesh_collider,
            model: MPMModelStruct,
        ):
            p = wp.tid()
            mesh = wp.mesh_get(param.mesh_id)

            vertex_index_0 = mesh.indices[3*p+0]
            vertex_index_1 = mesh.indices[3*p+1]
            vertex_index_2 = mesh.indices[3*p+2]

            vertex_point_0 = mesh.points[vertex_index_0]
            vertex_point_1 = mesh.points[vertex_index_1]
            vertex_point_2 = mesh.points[vertex_index_2]
            face_point = (vertex_point_0 + vertex_point_1 + vertex_point_2) / 3.0

            vertex_velocity_0 = mesh.velocities[vertex_index_0]
            vertex_velocity_1 = mesh.velocities[vertex_index_1]
            vertex_velocity_2 = mesh.velocities[vertex_index_2]
            face_velocity = (vertex_velocity_0 + vertex_velocity_1 + vertex_velocity_2) / 3.0

            face_normal = wp.mesh_eval_face_normal(param.mesh_id, p)

            grid_pos = face_point * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)

            if base_pos_x >= 0 and base_pos_x < model.grid_dim_x - 3 and base_pos_y >= 0 and base_pos_y < model.grid_dim_y - 3 and base_pos_z >= 0 and base_pos_z < model.grid_dim_z - 3:
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

                for i in range(0, 3):
                    for j in range(0, 3):
                        for k in range(0, 3):
                            ix = base_pos_x + i
                            iy = base_pos_y + j
                            iz = base_pos_z + k
                            weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                            wp.atomic_add(param.mesh_v_in, ix, iy, iz, weight * face_velocity)
                            wp.atomic_add(param.mesh_normal, ix, iy, iz, weight * face_normal)
                            wp.atomic_add(param.weight, ix, iy, iz, weight)
        
        @wp.kernel
        def normalize_grid(
            param: Mesh_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if param.weight[grid_x, grid_y, grid_z] > 1e-15:
                inv_w = 1.0 / param.weight[grid_x, grid_y, grid_z]
                v = param.mesh_v_in[grid_x, grid_y, grid_z] * inv_w
                param.mesh_v_out[grid_x, grid_y, grid_z] = v

        @wp.kernel
        def collide(
            state: MPMStateStruct,
            param: Mesh_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            v = state.grid_v_out[grid_x, grid_y, grid_z]
            if param.weight[grid_x, grid_y, grid_z] > 1e-15:
                v_rel = v - param.mesh_v_out[grid_x, grid_y, grid_z]
                n = wp.normalize(param.mesh_normal[grid_x, grid_y, grid_z])
                normal_component = wp.dot(v_rel, n)
                v_proj = (
                    v_rel - wp.min(normal_component, 0.0) * n
                )  # Project out only inward normal component
                if normal_component < 0.0 and wp.length(v_proj) > 1e-20:
                    v_fric = wp.max(
                        0.0, wp.length(v_proj) + normal_component * param.friction
                    ) * wp.normalize(
                        v_proj
                    )  # apply friction here
                    # ) + 1e-1 * n # apply friction here
                else:
                    v_fric = v_proj
                state.grid_v_out[grid_x, grid_y, grid_z] = v_fric + param.mesh_v_out[grid_x, grid_y, grid_z]
            else:
                state.grid_v_out[grid_x, grid_y, grid_z] = v
        self.mesh_colliders.append([zero_grid, compute_mesh, normalize_grid, collide])
        self.mesh_collider_params.append(collider_param)

    # a cubiod is a rectangular cube'
    # centered at `point`
    # dimension is x: point[0]±size[0]
    #              y: point[1]±size[1]
    #              z: point[2]±size[2]
    # all grid nodes lie within the cubiod will have their speed set to velocity
    # the cuboid itself is also moving with const speed = velocity
    # set the speed to zero to fix BC
    def set_velocity_on_cuboid(
        self,
        point,
        size,
        velocity,
        start_time=0.0,
        end_time=999.0,
        reset=0,
    ):
        point = list(point)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.size = size
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        # collider_param.threshold = threshold
        collider_param.reset = reset
        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if (
                    wp.abs(offset[0]) < param.size[0]
                    and wp.abs(offset[1]) < param.size[1]
                    and wp.abs(offset[2]) < param.size[2]
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = param.velocity
            elif param.reset == 1:
                if time < param.end_time + 15.0 * dt:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        def modify(time, dt, param: Dirichlet_collider):
            if time >= param.start_time and time < param.end_time:
                param.point = wp.vec3(
                    param.point[0] + dt * param.velocity[0],
                    param.point[1] + dt * param.velocity[1],
                    param.point[2] + dt * param.velocity[2],
                )  # param.point + dt * param.velocity

        self.grid_postprocess.append(collide)
        self.modify_bc.append(modify)

    def add_bounding_box(self, start_time=0.0, end_time=999.0):
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            padding = 3
            if time >= param.start_time and time < param.end_time:
                if grid_x < padding and state.grid_v_out[grid_x, grid_y, grid_z][0] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_x >= model.grid_dim_x - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][0] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                if grid_y < padding and state.grid_v_out[grid_x, grid_y, grid_z][1] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_y >= model.grid_dim_y - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][1] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                if grid_z < padding and state.grid_v_out[grid_x, grid_y, grid_z][2] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )
                if (
                    grid_z >= model.grid_dim_z - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][2] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # particle_v += force/particle_mass * dt
    # this is applied from start_dt, ends after num_dt p2g2p's
    # particle velocity is changed before p2g at each timestep
    def add_impulse_on_particles(
        self,
        mpm_state,
        force,
        dt,
        point=[1, 1, 1],
        size=[1, 1, 1],
        num_dt=1,
        start_time=0.0,
        device="cuda:0",
    ):
        impulse_param = Impulse_modifier()
        impulse_param.start_time = start_time
        impulse_param.end_time = start_time + dt * num_dt

        impulse_param.point = wp.vec3(point[0], point[1], point[2])
        impulse_param.size = wp.vec3(size[0], size[1], size[2])
        impulse_param.mask = wp.zeros(shape=self.n_particles, dtype=int, device=device)

        impulse_param.force = wp.vec3(
            force[0],
            force[1],
            force[2],
        )

        wp.launch(
            kernel=selection_add_impulse_on_particles,
            dim=self.n_particles,
            inputs=[mpm_state, impulse_param],
            device=device,
        )

        self.impulse_params.append(impulse_param)

        @wp.kernel
        def apply_force(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.mask[p] == 1:
                    impulse = wp.vec3(
                        param.force[0] / state.particle_mass[p],
                        param.force[1] / state.particle_mass[p],
                        param.force[2] / state.particle_mass[p],
                    )
                    state.particle_v[p] = state.particle_v[p] + impulse * dt

        self.pre_p2g_operations.append(apply_force)

    def enforce_particle_velocity_translation(
        self, mpm_state, point, size, velocity, start_time, end_time, device="cuda:0"
    ):
        # first select certain particles based on position

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.size = wp.vec3(size[0], size[1], size[2])

        velocity_modifier_params.velocity = wp.vec3(
            velocity[0], velocity[1], velocity[2]
        )

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )

        wp.launch(
            kernel=selection_enforce_particle_velocity_translation,
            dim=self.n_particles,
            inputs=[mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    state.particle_v[p] = velocity_modifier_params.velocity

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    # define a cylinder with center point, half_height, radius, normal
    # particles within the cylinder are rotating along the normal direction
    # may also have a translational velocity along the normal direction
    def enforce_particle_velocity_rotation(
        self,
        mpm_state,
        point,
        normal,
        half_height_and_radius,
        rotation_scale,
        translation_scale,
        start_time,
        end_time,
        device="cuda:0",
    ):
        normal_scale = 1.0 / wp.sqrt(
            float(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        )
        normal = list(normal_scale * x for x in normal)

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.half_height_and_radius = wp.vec2(
            half_height_and_radius[0], half_height_and_radius[1]
        )
        velocity_modifier_params.normal = wp.vec3(normal[0], normal[1], normal[2])

        horizontal_1 = wp.vec3(1.0, 1.0, 1.0)
        if wp.abs(wp.dot(velocity_modifier_params.normal, horizontal_1)) < 0.01:
            horizontal_1 = wp.vec3(0.72, 0.37, -0.67)
        horizontal_1 = (
            horizontal_1
            - wp.dot(horizontal_1, velocity_modifier_params.normal)
            * velocity_modifier_params.normal
        )
        horizontal_1 = horizontal_1 * (1.0 / wp.length(horizontal_1))
        horizontal_2 = wp.cross(horizontal_1, velocity_modifier_params.normal)

        velocity_modifier_params.horizontal_axis_1 = horizontal_1
        velocity_modifier_params.horizontal_axis_2 = horizontal_2

        velocity_modifier_params.rotation_scale = rotation_scale
        velocity_modifier_params.translation_scale = translation_scale

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )

        wp.launch(
            kernel=selection_enforce_particle_velocity_cylinder,
            dim=self.n_particles,
            inputs=[mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    offset = state.particle_x[p] - velocity_modifier_params.point
                    horizontal_distance = wp.length(
                        offset
                        - wp.dot(offset, velocity_modifier_params.normal)
                        * velocity_modifier_params.normal
                    )
                    cosine = (
                        wp.dot(offset, velocity_modifier_params.horizontal_axis_1)
                        / horizontal_distance
                    )
                    theta = wp.acos(cosine)
                    if wp.dot(offset, velocity_modifier_params.horizontal_axis_2) > 0:
                        theta = theta
                    else:
                        theta = -theta
                    axis1_scale = (
                        -horizontal_distance
                        * wp.sin(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis2_scale = (
                        horizontal_distance
                        * wp.cos(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis_vertical_scale = translation_scale
                    state.particle_v[p] = (
                        axis1_scale * velocity_modifier_params.horizontal_axis_1
                        + axis2_scale * velocity_modifier_params.horizontal_axis_2
                        + axis_vertical_scale * velocity_modifier_params.normal
                    )

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    # given normal direction, say [0,0,1]
    # gradually release grid velocities from start position to end position
    def release_particles_sequentially(
        self, mpm_state, normal, start_position, end_position, num_layers, start_time, end_time
    ):
        num_layers = 50
        point = [0, 0, 0]
        size = [0, 0, 0]
        axis = -1
        for i in range(3):
            if normal[i] == 0:
                point[i] = 1
                size[i] = 1
            else:
                axis = i
                point[i] = end_position

        half_length_portion = wp.abs(start_position - end_position) / num_layers
        end_time_portion = end_time / num_layers
        for i in range(num_layers):
            size[axis] = half_length_portion * (num_layers - i)
            self.enforce_particle_velocity_translation(
                mpm_state=mpm_state,
                point=point,
                size=size,
                velocity=[0, 0, 0],
                start_time=start_time,
                end_time=end_time_portion * (i + 1),
            )

    def enforce_particle_velocity_by_mask(
        self,
        mpm_state,
        selection_mask: torch.Tensor,
        velocity,
        start_time,
        end_time,
    ):
        # first select certain particles based on position

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.velocity = wp.vec3(
            velocity[0],
            velocity[1],
            velocity[2],
        )

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.from_torch(selection_mask)

        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    state.particle_v[p] = velocity_modifier_params.velocity

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    def enforce_grid_velocity_by_mask(
        self,
        selection_mask: torch.Tensor,  # should be int
    ):

        grid_modifier_params = GridCollider()

        grid_modifier_params.mask = wp.from_torch(selection_mask)

        self.collider_params.append(grid_modifier_params)

        @wp.kernel
        def modify_grid_v_before_g2p(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            grid_modifier_params: GridCollider,
        ):
            grid_x, grid_y, grid_z = wp.tid()

            if grid_modifier_params.mask[grid_x, grid_y, grid_z] >= 1:
                state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        self.grid_postprocess.append(modify_grid_v_before_g2p)
        self.modify_bc.append(None)

    # particle_v += force/particle_mass * dt
    # this is applied from start_dt, ends after num_dt p2g2p's
    # particle velocity is changed before p2g at each timestep
    def add_impulse_on_particles_with_mask(
        self,
        mpm_state,
        force,
        dt,
        particle_mask,  # 1 for selected particles, 0 for others
        point=[1, 1, 1],
        size=[1, 1, 1],
        end_time=1,
        start_time=0.0,
        device="cuda:0",
    ):
        assert (
            len(particle_mask) == self.n_particles
        ), "mask should have n_particles elements"
        impulse_param = Impulse_modifier()
        impulse_param.start_time = start_time
        impulse_param.end_time = end_time
        impulse_param.mask = wp.from_torch(particle_mask)

        impulse_param.point = wp.vec3(point[0], point[1], point[2])
        impulse_param.size = wp.vec3(size[0], size[1], size[2])

        impulse_param.force = wp.vec3(
            force[0],
            force[1],
            force[2],
        )

        wp.launch(
            kernel=selection_add_impulse_on_particles,
            dim=self.n_particles,
            inputs=[mpm_state, impulse_param],
            device=device,
        )

        self.impulse_params.append(impulse_param)

        @wp.kernel
        def apply_force(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.mask[p] >= 1:
                    # impulse = wp.vec3(
                    #     param.force[0] / state.particle_mass[p],
                    #     param.force[1] / state.particle_mass[p],
                    #     param.force[2] / state.particle_mass[p],
                    # )
                    impulse = wp.vec3(
                        param.force[0],
                        param.force[1],
                        param.force[2],
                    )
                    state.particle_v[p] = state.particle_v[p] + impulse * dt

        self.pre_p2g_operations.append(apply_force)
