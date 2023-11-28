import pickle
from holonomic_research.biorbd_model_holonomic_updated import BiorbdModelCustomHolonomic

def save_results(sol, c3d_file_path, q_complete, qdot_complete, qddot_complete, lambdas_complete):
    """
    Solving the ocp
    Parameters
     ----------
     sol: Solution
        The solution to the ocp at the current pool
    c3d_file_path: str
        The path to the c3d file of the task
    """

    data = {}
    data["q_complete"] = q_complete
    data["qdot_complete"] = qdot_complete
    data["qddot_complete"] = qddot_complete
    data["lambdas_complete"] = lambdas_complete

    q = []
    qdot = []
    states_all = []
    tau = []

    if len(sol.ns) == 1:
        q = sol.states["q_u"]
        qdot = sol.states["q_udot"]
        # states_all = sol.states["all"]
        tau = sol.controls["tau"]
    else:
        for i in range(len(sol.states)):
            if i == 1:
                q.append(sol.states[i]["q_u"])
                qdot.append(sol.states[i]["qdot_u"])
                # states_all.append(sol.states[i]["all"])
                tau.append(sol.controls[i]["tau"])
            else:
                q.append(sol.states[i]["q"])
                qdot.append(sol.states[i]["qdot"])
                # states_all.append(sol.states[i]["all"])
                tau.append(sol.controls[i]["tau"])

    data["q"] = q
    data["qdot"] = qdot
    data["tau"] = tau
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    # data["detailed_cost"] = sol.detailed_cost
    data["status"] = sol.status
    data["real_time_to_optimize"] = sol.real_time_to_optimize
    data["phase_time"] = sol.phase_time[1:12]
    data["constraints"] = sol.constraints
    data["controls"] = sol.controls
    data["constraints_scaled"] = sol.controls_scaled
    data["n_shooting"] = sol.ns
    data["time"] = sol.time
    data["lam_g"] = sol.lam_g
    data["lam_p"] = sol.lam_p
    data["lam_x"] = sol.lam_x

    if sol.status == 1:
        data["status"] = "Optimal Control Solution Found"
    else:
        data["status"] = "Restoration Failed !"

    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


def compute_all_states(sol, bio_model: BiorbdModelCustomHolonomic, index_holonomics_constraints: int):
    """
    Compute all the states from the solution of the optimal control program

    Parameters
    ----------

    sol:
        The solution of the optimal control program
    bio_model: HolonomicBiorbdModel
        The biorbd model

    Returns
    -------

    """
    n = sol.states[index_holonomics_constraints]["q_u"].shape[1]
    nb_root = bio_model.nb_root
    q = np.zeros((bio_model.nb_q, n))
    qdot = np.zeros((bio_model.nb_q, n))
    qddot = np.zeros((bio_model.nb_q, n))
    lambdas = np.zeros((bio_model.nb_dependent_joints, n))
    tau = np.ones((bio_model.nb_tau, n))
    tau_independent = [element - 3 for element in bio_model.independent_joint_index[3:]]
    tau_dependent = [element - 3 for element in bio_model.dependent_joint_index]

    for i, independent_joint_index in enumerate(bio_model.independent_joint_index[3:]):
        tau[independent_joint_index] = sol.controls[index_holonomics_constraints]["tau"][tau_independent[i], :]
    for i, dependent_joint_index in enumerate(bio_model.dependent_joint_index):
        tau[dependent_joint_index] = sol.controls[index_holonomics_constraints]["tau"][tau_dependent[i], :]

    # Partitioned forward dynamics
    q_u_sym = MX.sym("q_u_sym", bio_model.nb_independent_joints, 1)
    qdot_u_sym = MX.sym("qdot_u_sym", bio_model.nb_independent_joints, 1)
    tau_sym = MX.sym("tau_sym", bio_model.nb_tau, 1)
    partitioned_forward_dynamics_func = Function(
        "partitioned_forward_dynamics",
        [q_u_sym, qdot_u_sym, tau_sym],
        [bio_model.partitioned_forward_dynamics(q_u_sym, qdot_u_sym, tau_sym)],
    )
    # Lagrangian multipliers
    q_sym = MX.sym("q_sym", bio_model.nb_q, 1)
    qdot_sym = MX.sym("qdot_sym", bio_model.nb_q, 1)
    qddot_sym = MX.sym("qddot_sym", bio_model.nb_q, 1)
    compute_lambdas_func = Function(
        "compute_the_lagrangian_multipliers",
        [q_sym, qdot_sym, qddot_sym, tau_sym],
        [bio_model.compute_the_lagrangian_multipliers(q_sym, qdot_sym, qddot_sym, tau_sym)],
    )

    for i in range(n):
        q_v_i = bio_model.compute_v_from_u_explicit_numeric(
            sol.states[index_holonomics_constraints]["q_u"][:, i]
        ).toarray()
        q[:, i] = (
            bio_model.state_from_partition(sol.states[index_holonomics_constraints]["q_u"][:, i][:, np.newaxis], q_v_i)
            .toarray()
            .squeeze()
        )
        qdot[:, i] = (
            bio_model.compute_qdot(q[:, i], sol.states[index_holonomics_constraints]["qdot_u"][:, i])
            .toarray()
            .squeeze()
        )
        qddot_u_i = (
            partitioned_forward_dynamics_func(
                sol.states[index_holonomics_constraints]["q_u"][:, i],
                sol.states[index_holonomics_constraints]["qdot_u"][:, i],
                tau[:, i],
            )
            .toarray()
            .squeeze()
        )
        qddot[:, i] = bio_model.compute_qddot(q[:, i], qdot[:, i], qddot_u_i).toarray().squeeze()
        lambdas[:, i] = (
            compute_lambdas_func(
                q[:, i],
                qdot[:, i],
                qddot[:, i],
                tau[:, i],
            )
            .toarray()
            .squeeze()
        )

    return q, qdot, qddot, lambdas
