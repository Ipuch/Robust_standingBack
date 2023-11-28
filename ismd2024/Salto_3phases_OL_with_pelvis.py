"""
The aim of this code is to test the holonomic constraint of the flight phase
with the pelvis during the flight phase (no holonomic constraints),
the tucked phase (holonomic constraints) and
the preparation of landing (no holonomic constraints).
We also want to see how well the transition
between phases with and without holonomic constraints works.

Phase 0: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 1: Tucked phase
- Dynamic(s): TORQUE_DRIVEN with holonomic constraints
- Constraint(s): zero contact, 1 holonomic constraints body-body
- Objective(s) function(s): minimize torque and time

Phase 2: Preparation landing
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints body-body
- Objective(s) function(s): minimize torque and time

"""
# --- Import package --- #
import matplotlib.pyplot as plt
import numpy as np
import pickle

# import matplotlib.pyplot as plt
from bioptim import (
    BiorbdModel,
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    Solver,
    HolonomicConstraintsList,
    HolonomicConstraintsFcn,
    HolonomicBiorbdModel,
    PenaltyController,
    QuadratureRule,
    DynamicsFunctions,
)

from casadi import MX, vertcat, Function
from holonomic_research.biorbd_model_holonomic_updated import BiorbdModelCustomHolonomic
from visualisation import visualisation_closed_loop_3phases


# --- Save results --- #
def save_results(sol, c3d_file_path):
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


def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break

    return data_tmp


# --- Parameters --- #
movement = "Salto_open_loop"
version = 19
nb_phase = 3
name_folder_model = "../model"
pickle_sol_init = (
    "Salto_CL_3phases_with_pelvis_V17.pkl"
)


sol = get_created_data_from_pickle(pickle_sol_init)
# q_init_holonomic = sol["q"][index_holonomics_constraints][independent_joint_index]
# qdot_init_holonomic = sol["qdot"][index_holonomics_constraints][independent_joint_index]

phase_time_init = []
for i in range(len(sol["time"])):
    time_final = sol["time"][i][-1] - sol["time"][i][0]
    phase_time_init.append(time_final)

n_shooting_init = []
for i in range(len(sol["time"])):
    n_shooting_final = sol["time"][i].shape[0] - 1
    n_shooting_init.append(n_shooting_final)

# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting) -> (HolonomicBiorbdModel, OptimalControlProgram):
    bio_model = (
        BiorbdModel(biorbd_model_path[0]),
        BiorbdModel(biorbd_model_path[1]),
        BiorbdModel(biorbd_model_path[2]),
    )

    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.9 for i in tau_min_total]
    tau_max = [i * 0.9 for i in tau_max_total]
    tau_init = 0
    variable_bimapping = BiMappingList()
    dof_mapping = BiMappingList()
    variable_bimapping.add("q", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    variable_bimapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Flight phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=0, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=0)

    # Phase 1 (Tucked phase:
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=1, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.01, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.01, phase=1)

    # Phase 2 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, phase=2, min_bound=0.1, max_bound=0.3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=2)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0, expand_dynamics = True)
    dynamics.add(
        bio_model[1].holonomic_torque_driven,
        dynamic_function=DynamicsFunctions.holonomic_torque_driven,
        mapping=variable_bimapping,
        expand_dynamics = True,
    )
    # dynamics.add(DynamicsFcn.HOLONOMIC_TORQUE_DRIVEN, expand=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2, expand_dynamics=True)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_phase_transition_pre, phase_pre_idx=0)
    phase_transitions.add(custom_phase_transition_post, phase_pre_idx=1)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()
    holonomic_constraints = HolonomicConstraintsList()

    # Phase 0: Take-off
    holonomic_constraints.add(
        "holonomic_constraints",
        HolonomicConstraintsFcn.superimpose_markers,
        biorbd_model=bio_model[1],
        marker_1="BELOW_KNEE",
        marker_2="CENTER_HAND",
        index=slice(1, 3),
        local_frame_index=11,
    )
    # Made up constraints

    bio_model[1].set_holonomic_configuration(
        constraints_list=holonomic_constraints,
        independent_joint_index=[0, 1, 2, 5, 6, 7],
        dependent_joint_index=[3, 4],
    )

    # Path constraint

    pose_takeout_start = [-0.2777, 0.0399, 0.1930, 2.5896, 0.51, 0.5354, -0.8367, 0.1119]
    pose_salto_start = [-0.6369, 1.0356, 1.5062, 0.3411, 1.3528, 2.1667, -1.9179, 0.0393]
    pose_salto_end = [0.1987, 1.0356, 2.7470, 0.9906, 0.0252, 1.7447, -1.1335, 0.0097]
    pose_salto_start_CL = [-0.6369, 1.0356, 1.5062, 2.1667, -1.9179, 0.0393]
    pose_salto_end_CL = [0.1987, 1.0356, 2.7470, 1.7447, -1.1335, 0.0097]
    pose_landing_start = [0.1987, 1.7551, 5.8322, 0.52, 0.95, 1.72, -0.81, 0.0]

    # --- Bounds ---#
    # Initialize x_bounds
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    n_independent = bio_model[1].nb_independent_joints

    # Phase 0: Flight phase
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_takeout_start
    x_bounds[0]["q"].min[0, :] = -1
    x_bounds[0]["q"].max[0, :] = 1
    x_bounds[0]["q"].min[1, 1:] = 0
    x_bounds[0]["q"].max[1, 1:] = 2.5
    x_bounds[0]["q"].min[2, 1] = -np.pi / 4
    x_bounds[0]["q"].max[2, 1] = np.pi / 2
    x_bounds[0]["q"].min[2, -1] = -np.pi / 2
    x_bounds[0]["q"].max[2, -1] = np.pi / 2
    x_bounds[0]["q"].min[4, -1] = 1

    x_bounds[0]["qdot"].min[0, :] = -5
    x_bounds[0]["qdot"].max[0, :] = 5
    x_bounds[0]["qdot"].min[1, :] = -2
    x_bounds[0]["qdot"].max[1, :] = 10
    x_bounds[0]["qdot"].min[2, :] = -5
    x_bounds[0]["qdot"].max[2, :] = 5

    # Phase 1: Tucked phase
    x_bounds.add("q_u", bounds=bio_model[1].bounds_from_ranges("q", mapping=variable_bimapping), phase=1)
    x_bounds.add("qdot_u", bounds=bio_model[1].bounds_from_ranges("qdot", mapping=variable_bimapping), phase=1)
    x_bounds[1]["q_u"].min[0, :] = -2
    x_bounds[1]["q_u"].max[0, :] = 1
    x_bounds[1]["q_u"].min[1, 1:] = 0
    x_bounds[1]["q_u"].max[1, 1:] = 2.5
    x_bounds[1]["q_u"].min[2, 0] = 0
    x_bounds[1]["q_u"].max[2, 0] = np.pi / 2
    x_bounds[1]["q_u"].min[2, 1] = np.pi / 8
    x_bounds[1]["q_u"].max[2, 1] = 2 * np.pi
    x_bounds[1]["q_u"].min[2, 2] = 3 / 4 * np.pi
    x_bounds[1]["q_u"].max[2, 2] = 3 / 2 * np.pi
    x_bounds[1]["qdot_u"].min[0, :] = -5
    x_bounds[1]["qdot_u"].max[0, :] = 5
    x_bounds[1]["qdot_u"].min[1, :] = -2
    x_bounds[1]["qdot_u"].max[1, :] = 10
    x_bounds[1]["q_u"].max[3, :] = 2.6
    x_bounds[1]["q_u"].min[3, :] = 1.30

    # Phase 2: Preparation landing
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[0, :] = -2
    x_bounds[2]["q"].max[0, :] = 1
    x_bounds[2]["q"].min[1, 1:] = 0
    x_bounds[2]["q"].max[1, 1:] = 2.5
    x_bounds[2]["q"].min[2, :] = 3 / 4 * np.pi
    x_bounds[2]["q"].max[2, :] = 2 * np.pi + 0.5
    x_bounds[2]["qdot"].min[0, :] = -5
    x_bounds[2]["qdot"].max[0, :] = 5
    x_bounds[2]["qdot"].min[1, :] = -10
    x_bounds[2]["qdot"].max[1, :] = 10
    x_bounds[2]["q"].max[:, -1] = np.array(pose_landing_start) + 0.5
    x_bounds[2]["q"].min[:, -1] = np.array(pose_landing_start) - 0.5

    # Initial guess
    x_init = InitialGuessList()
    # x_init.add("q", np.array([pose_takeout_start, pose_salto_start]).T, interpolation=InterpolationType.LINEAR, phase=0)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q_u", sol["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q_udot", sol["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)

    # x_init.add("q_u", np.array([pose_salto_start_CL, pose_salto_end_CL]).T, interpolation=InterpolationType.LINEAR,
    #            phase=1)
    # x_init.add("q_udot", np.array([[0] * n_independent, [0] * n_independent]).T, interpolation=InterpolationType.LINEAR,
    #            phase=1)
    x_init.add("q_u", sol["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("q_udot", sol["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)

    # x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR, phase=2)
    # x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("q", sol["q"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    x_init.add("qdot", sol["qdot"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=0,
    )
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=1,
    )
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=2,
    )

    u_init = InitialGuessList()
    # u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    # u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=1)
    # u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)
    u_init.add("tau", sol["tau"][0][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=0)
    u_init.add("tau", sol["tau"][1][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    u_init.add("tau", sol["tau"][2][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=2)

    return (
        OptimalControlProgram(
            bio_model=bio_model,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=phase_time,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            constraints=constraints,
            n_threads=32,
            phase_transitions=phase_transitions,
            variable_mappings=dof_mapping,
        ),
        bio_model,
    )


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V2.bioMod"
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path, model_path, model_path),
        phase_time=(
            phase_time_init[0],
            phase_time_init[1],
            phase_time_init[2],
        ),
        n_shooting=(
            n_shooting_init[0],
            n_shooting_init[1],
            n_shooting_init[2],
        ),
    )

    # ocp.add_plot_penalty()
    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    # solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)

    sol = ocp.solve(solver)

    # --- Show results --- #
    save_results(sol, str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    visualisation_closed_loop_3phases(bio_model, sol, model_path)
    # sol.graphs(show_bounds=True)

    plt.show()


if __name__ == "__main__":
    main()
