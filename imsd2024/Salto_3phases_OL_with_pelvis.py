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
    ConstraintFcn,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    PhaseTransitionFcn,
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
    Node,
)

from utils import save_results, compute_all_states
from save import get_created_data_from_pickle
from visualisation import visualisation_closed_loop_3phases

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

    dof_mapping = BiMappingList()
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
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0, expand_dynamics=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=1, expand_dynamics=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2, expand_dynamics=True)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.ALL,
                    first_marker="CENTER_HAND",
                    second_marker="BELOW_KNEE",
                    )

    pose_takeout_start = [-0.2777, 0.0399, 0.1930, 2.5896, 0.51, 0.5354, -0.8367, 0.1119]
    pose_salto_start = [-0.6369, 1.0356, 1.5062, 0.3411, 1.3528, 2.1667, -1.9179, 0.0393]
    pose_salto_end = [0.1987, 1.0356, 2.7470, 0.9906, 0.0252, 1.7447, -1.1335, 0.0097]
    pose_salto_start_CL = [-0.6369, 1.0356, 1.5062, 2.1667, -1.9179, 0.0393]
    pose_salto_end_CL = [0.1987, 1.0356, 2.7470, 1.7447, -1.1335, 0.0097]
    pose_landing_start = [0.1987, 1.7551, 5.8322, 0.52, 0.95, 1.72, -0.81, 0.0]

    # --- Bounds ---#
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
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[0, :] = -2
    x_bounds[1]["q"].max[0, :] = 1
    x_bounds[1]["q"].min[1, 1:] = 0
    x_bounds[1]["q"].max[1, 1:] = 2.5
    x_bounds[1]["q"].min[2, 0] = 0
    x_bounds[1]["q"].max[2, 0] = np.pi / 2
    x_bounds[1]["q"].min[2, 1] = np.pi / 8
    x_bounds[1]["q"].max[2, 1] = 2 * np.pi
    x_bounds[1]["q"].min[2, 2] = 3 / 4 * np.pi
    x_bounds[1]["q"].max[2, 2] = 3 / 2 * np.pi
    x_bounds[1]["qdot"].min[0, :] = -5
    x_bounds[1]["qdot"].max[0, :] = 5
    x_bounds[1]["qdot"].min[1, :] = -2
    x_bounds[1]["qdot"].max[1, :] = 10
    x_bounds[1]["q"].max[3, :] = 2.6
    x_bounds[1]["q"].min[3, :] = 1.30
    x_bounds[1]["q"].max[5, :] = 0.1
    x_bounds[1]["q"].min[5, :] = -0.7

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
    x_init.add("q", sol["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q", sol["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)

    # x_init.add("q_u", np.array([pose_salto_start_CL, pose_salto_end_CL]).T, interpolation=InterpolationType.LINEAR,
    #            phase=1)
    # x_init.add("q_udot", np.array([[0] * n_independent, [0] * n_independent]).T, interpolation=InterpolationType.LINEAR,
    #            phase=1)

    # insert two row of zeros in index 3 and 4 in sol["q"][1] and sol["qdot"][1]
    sol["q"][1] = np.insert(sol["q"][1], 3, np.zeros((2, sol["q"][1].shape[1])), axis=0)
    sol["qdot"][1] = np.insert(sol["qdot"][1], 3, np.zeros((2, sol["qdot"][1].shape[1])), axis=0)

    x_init.add("q", sol["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", sol["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)

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
        phase_time=(0.30000000999781434, 0.3000000099876983, 0.3000000099830904),
        n_shooting=(20, 30, 20),
    )

    # ocp.add_plot_penalty()
    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    # solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(100)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)

    sol = ocp.solve(solver)

    # --- Show results --- #
    save_results(
        sol,
        str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl",
        None,
        None,
        None,
        None,
    )
    # visualisation_closed_loop_3phases(bio_model, sol, model_path)
    sol.graphs(show_bounds=True)
    # plt.show()


if __name__ == "__main__":
    main()
