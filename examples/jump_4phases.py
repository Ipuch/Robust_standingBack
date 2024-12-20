"""
The aim of this code is to create a movement a simple jump in 3 phases with a 2D model.

Phase 0: Propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): one contact (toe)
- Objective(s) function(s): maximize velocity of CoM and minimize time of flight

Phase 1: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact (in the air)
- Objective(s) function(s): maximize height of CoM and maximize time of flight

Phase 2: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact (in the air)
- Objective(s) function(s): maximize height of CoM and maximize time of flight

Phase 3: Landing
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): two contact
- Objective(s) function(s): minimize velocity CoM and minimize state qdot
"""

# --- Import package --- #
import numpy as np
import pickle
from bioptim import (
    BiorbdModel,
    Node,
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    DynamicsFcn,
    BiMappingList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    Solver,
    Axis,
    SolutionMerge,
    PhaseTransitionFcn,
)
from src.save_load_helpers import get_created_data_from_pickle
from src.constraints import CoM_over_toes


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
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    list_time = sol.decision_time(to_merge=SolutionMerge.NODES)

    q = []
    qdot = []
    qddot = []
    tau = []
    time = []
    min_bounds_q = []
    max_bounds_q = []
    min_bounds_qdot = []
    max_bounds_qdot = []
    min_bounds_tau = []
    max_bounds_tau = []

    for i in range(len(states)):
        q.append(states[i]["q"])
        qdot.append(states[i]["qdot"])
        tau.append(controls[i]["tau"])
        time.append(list_time[i])
        min_bounds_q.append(sol.ocp.nlp[i].x_bounds["q"].min)
        max_bounds_q.append(sol.ocp.nlp[i].x_bounds["q"].max)
        min_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot"].min)
        max_bounds_qdot.append(sol.ocp.nlp[i].x_bounds["qdot"].max)
        min_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].min)
        max_bounds_tau.append(sol.ocp.nlp[i].u_bounds["tau"].max)

    data["q"] = q
    data["qdot"] = qdot
    data["tau"] = tau
    data["time"] = time
    data["min_bounds_q"] = min_bounds_q
    data["max_bounds_q"] = max_bounds_q
    data["min_bounds_qdot"] = min_bounds_qdot
    data["max_bounds_qdot"] = max_bounds_qdot
    data["min_bounds_tau"] = min_bounds_q
    data["max_bounds_tau"] = max_bounds_q
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    # data["detailed_cost"] = sol.add_detailed_cost
    data["status"] = sol.status
    data["real_time_to_optimize"] = sol.real_time_to_optimize
    data["phase_time"] = sol.phases_dt
    data["constraints"] = sol.constraints
    data["n_shooting"] = sol.ocp.n_shooting
    data["lam_g"] = sol.lam_g
    data["lam_p"] = sol.lam_p
    data["lam_x"] = sol.lam_x
    data["phase_time"] = sol.ocp.phase_time
    data["dof_names"] = sol.ocp.nlp[0].dof_names
    data["q_all"] = np.hstack(data["q"])
    data["qdot_all"] = np.hstack(data["qdot"])
    data["tau_all"] = np.hstack(data["tau"])
    time_end_phase = []
    time_total = 0
    time_all = []
    for i in range(len(data["time"])):
        time_all.append(data["time"][i] + time_total)
        time_total = time_total + data["time"][i][-1]
        time_end_phase.append(time_total)
    data["time_all"] = np.vstack(time_all)
    data["time_total"] = time_total
    data["time_end_phase"] = time_end_phase

    if sol.status == 1:
        data["status"] = "Optimal Control Solution Found"
    else:
        data["status"] = "Restoration Failed !"

    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


# --- Parameters --- #
movement = "jump"
version = 22
nb_phase = 4
name_folder_model = "../models"
pickle_sol_init = "../jump_4phases_V20.pkl"
sol_jump = get_created_data_from_pickle(pickle_sol_init)


# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):

    # --- Options --- #
    # BioModel path
    bio_model = (
        BiorbdModel(biorbd_model_path[0]),
        BiorbdModel(biorbd_model_path[1]),
        BiorbdModel(biorbd_model_path[2]),
        BiorbdModel(biorbd_model_path[3]),
    )
    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.7 for i in tau_min_total]
    tau_max = [i * 0.7 for i in tau_max_total]
    tau_init = 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Propulsion):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, axes=Axis.Z, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1000, min_bound=0.01, max_bound=0.2, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.0001, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.0001, phase=0)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTACT_FORCES_END_OF_INTERVAL,
        node=Node.PENULTIMATE,
        weight=0.01,
        contact_index=0,
        quadratic=True,
        phase=0,
    )

    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_CONTACT_FORCES_END_OF_INTERVAL,
        node=Node.PENULTIMATE,
        weight=0.01,
        contact_index=1,
        quadratic=True,
        phase=0,
    )

    # Phase 1 (Flight):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=1)

    # Phase 2 (Preparation landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, phase=2)

    # Phase 3 (Landing):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, phase=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, min_bound=0.1, max_bound=0.3, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=100, axes=Axis.Y, phase=3)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=3)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=2)

    # Constraints
    # Phase 0: Propulsion
    constraints = ConstraintList()
    # Phase 0 (Propulsion):
    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Z,
        max_bound=0,
        min_bound=0,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Y,
        max_bound=0.1,
        min_bound=-0.1,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        CoM_over_toes,
        node=Node.START,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
        phase=0,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.01,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=0,
    )

    # constraints.add(
    #    ConstraintFcn.TRACK_CONTACT_FORCES_END_OF_INTERVAL,
    #    node=Node.PENULTIMATE,
    #    contact_index=1,
    #    quadratic=True,
    #    phase=0,
    # )

    # Phase 3 (Landing):

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0.01,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=3,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Z,
        max_bound=0,
        min_bound=0,
        node=Node.END,
        phase=3,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        marker_index="Foot_Toe_marker",
        axes=Axis.Y,
        max_bound=0.1,
        min_bound=-0.1,
        node=Node.END,
        phase=3,
    )

    constraints.add(
        CoM_over_toes,
        node=Node.END,
        phase=3,
    )

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.5,
        phase=3,
    )
    # Path constraint
    n_q = bio_model[0].nb_q
    n_qdot = n_q

    # Contraint position
    pose_propulsion_start = [0.0, -0.17, -0.9124, 0.0, 0.1936, 2.0082, -1.7997, 0.6472]
    pose_takeout_start = [0, 0, 0, 2.5896, 0.51, 0.5354, -0.8367, 0.1119]
    pose_tuck = [0, 1, 0.17, 0.3411, 1.3528, 2.1667, -1.9179, 0.0393]
    pose_landing_end = [0, 0, 0.1930, 3.1, 0.03, 0.0, 0.0, 0.0]

    # Initialize x_bounds
    x_bounds = BoundsList()

    # Phase 0: Propulsion
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"].min[2:7, 0] = np.array(pose_propulsion_start[2:7]) - 0.2
    x_bounds[0]["q"].max[2:7, 0] = np.array(pose_propulsion_start[2:7]) + 0.2
    x_bounds[0]["qdot"][:, 0] = [0] * n_qdot

    x_bounds[0]["q"].min[0, :] = -1
    x_bounds[0]["q"].max[0, :] = 1
    x_bounds[0]["qdot"].min[3, :] = 0

    # Phase 1: Flight
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds[1]["q"].min[0, :] = -1
    x_bounds[1]["q"].max[0, :] = 1
    x_bounds[1]["q"].min[1, -1] = 0.5
    x_bounds[1]["q"].max[1, -1] = 3
    x_bounds[1]["q"].min[2:, -1] = np.array(pose_tuck[2:]) - 0.2

    # Phase 2: Second Flight
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[0, :] = -1
    x_bounds[2]["q"].max[0, :] = 1
    x_bounds[2]["q"].min[1, 0] = 0.5
    x_bounds[2]["q"].max[1, 0] = 3

    # Phase 2: Landing
    x_bounds.add("q", bounds=bio_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=bio_model[3].bounds_from_ranges("qdot"), phase=3)
    x_bounds[3]["q"].max[2:7, 2] = np.array(pose_landing_end[2:7]) + 0.3  # 0.5
    x_bounds[3]["q"].min[2:7, 2] = np.array(pose_landing_end[2:7]) - 0.3

    x_bounds[3]["q"].min[0, :] = -1
    x_bounds[3]["q"].max[0, :] = 1

    # Initial guess
    x_init = InitialGuessList()

    x_init.add("q", sol_jump["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", sol_jump["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q", sol_jump["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", sol_jump["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("q", sol_jump["q"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    x_init.add("qdot", sol_jump["qdot"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    x_init.add("q", sol_jump["q"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)
    x_init.add("qdot", sol_jump["qdot"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)

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
    u_bounds.add(
        "tau",
        min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
        max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
        phase=3,
    )

    u_init = InitialGuessList()

    u_init.add("tau", sol_jump["tau"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    u_init.add("tau", sol_jump["tau"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    u_init.add("tau", sol_jump["tau"][2], interpolation=InterpolationType.EACH_FRAME, phase=2)
    u_init.add("tau", sol_jump["tau"][3], interpolation=InterpolationType.EACH_FRAME, phase=3)

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
            # assume_phase_dynamics=True,
            phase_transitions=phase_transitions,
            variable_mappings=dof_mapping,
        ),
        bio_model,
    )


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V3.bioMod"
    model_path_1contact = str(name_folder_model) + "/" + "Model2D_7Dof_2C_5M_CL_V3.bioMod"

    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path_1contact, model_path, model_path, model_path_1contact),
        phase_time=(0.2, 0.2, 0.2, 0.3),
        n_shooting=(20, 20, 20, 30),
        min_bound=0.01,
        max_bound=np.inf,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_options=dict(show_bounds=True), _linear_solver="MA57")  # show_online_optim=True,
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)

    sol = ocp.solve(solver)
    sol.print_cost()

    # --- Show results --- #
    save_results(sol, str(movement) + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    sol.graphs(show_bounds=True, save_name=str(movement) + "_" + str(nb_phase) + "phases_V" + str(version))


if __name__ == "__main__":
    main()
