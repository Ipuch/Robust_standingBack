"""
The aim of this code is to test the holonomic constraint of the flight phase
with the pelvis and with the landing phase.
The simulation have 5 phases: propulsion, flight phase, tucked phase, preparation landing, landing.
We also want to see how well the transition between phases with and without holonomic constraints works.

Phase 0: Propulsion
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 1 contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 1: Flight
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 2: Tucked phase
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact,
- Objective(s) function(s): minimize torque and time

Phase 3: Preparation landing
- Dynamic(s): TORQUE_DRIVEN
- Constraint(s): zero contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time

Phase 4: Landing
- Dynamic(s): TORQUE_DRIVEN with contact
- Constraint(s): 2 contact, no holonomic constraints
- Objective(s) function(s): minimize torque and time
"""

# --- Import package --- #
import numpy as np
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
    PhaseTransitionFcn,
    MagnitudeType,
)

from src.actuator_constants import ACTUATORS, initialize_tau
from src.bounds_x import add_x_bounds
from src.constants import (
    POSE_TUCKING_START,
    POSE_TUCKING_END,
    POSE_LANDING_START,
    JUMP_INIT_PATH,
    PATH_MODEL,
    PATH_MODEL_1_CONTACT,
)
from src.constraints import add_constraints
from src.multistart import prepare_multi_start
from src.objectives import minimize_actuator_torques, add_objectives, add_tau_derivative_objectives
from src.save_results import save_results
from src.save_load_helpers import get_created_data_from_pickle


def add_u_bounds(u_bounds, tau_min, tau_max):
    for i in range(5):
        u_bounds.add(
            "tau",
            min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
            max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]],
            phase=i,
        )

    return u_bounds


# --- Prepare ocp --- #
def prepare_ocp(biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START, seed=0):
    bio_model = (
        BiorbdModel(biorbd_model_path[0]),
        BiorbdModel(biorbd_model_path[1]),
        BiorbdModel(biorbd_model_path[2]),
        BiorbdModel(biorbd_model_path[3]),
        BiorbdModel(biorbd_model_path[4]),
    )

    n_q = bio_model[0].nb_q
    n_qdot = n_q

    # Actuators parameters
    actuators = ACTUATORS

    tau_min, tau_max, tau_init = initialize_tau()
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions = add_objectives(objective_functions, actuators)
    objective_functions = add_tau_derivative_objectives(objective_functions)
    objective_functions.add(
        minimize_actuator_torques,
        custom_type=ObjectiveFcn.Lagrange,
        actuators=actuators,
        quadratic=True,
        weight=0.1,
        phase=2,
    )

    # --- Dynamics ---#
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=0)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, phase=3)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, expand_continuity=False, with_contact=True, phase=4)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()
    constraints = add_constraints(constraints)

    # --- Bounds ---#
    x_bounds = BoundsList()
    q_bounds, qdot_bounds = add_x_bounds(bio_model)
    for i_phase in range(len(bio_model)):
        x_bounds.add("q", bounds=q_bounds[i_phase], phase=i_phase)
        x_bounds.add("qdot", bounds=qdot_bounds[i_phase], phase=i_phase)

    # Initial guess
    sol_salto = get_created_data_from_pickle(JUMP_INIT_PATH)
    x_init = InitialGuessList()
    # Initial guess from Jump
    x_init.add("q", sol_salto["q"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", sol_salto["qdot"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q", sol_salto["q"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", sol_salto["qdot"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("q", np.array([POSE_TUCKING_START, POSE_TUCKING_END]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("q", np.array([POSE_TUCKING_END, POSE_LANDING_START]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("q", sol_salto["q"][3], interpolation=InterpolationType.EACH_FRAME, phase=4)
    x_init.add("qdot", sol_salto["qdot"][3], interpolation=InterpolationType.EACH_FRAME, phase=4)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds = add_u_bounds(u_bounds, tau_min, tau_max)

    u_init = InitialGuessList()
    # Initial guess from jump
    u_init.add("tau", sol_salto["tau"][0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    u_init.add("tau", sol_salto["tau"][1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=3)
    u_init.add("tau", sol_salto["tau"][3], interpolation=InterpolationType.EACH_FRAME, phase=4)

    if WITH_MULTI_START:
        x_init.add_noise(
            bounds=x_bounds,
            magnitude=0.2,
            magnitude_type=MagnitudeType.RELATIVE,
            n_shooting=[n_shooting[i] + 1 for i in range(len(n_shooting))],
            seed=seed,
        )
        u_init.add_noise(
            bounds=u_bounds,
            magnitude=0.2,
            magnitude_type=MagnitudeType.RELATIVE,
            n_shooting=[n_shooting[i] for i in range(len(n_shooting))],
            seed=seed,
        )

    return OptimalControlProgram(
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
    )


def main():
    movement = "Salto"
    version = "Eve_final3"

    WITH_MULTI_START = False
    save_folder = f"../results/{str(movement)}_V{version}"

    biorbd_model_path = (PATH_MODEL_1_CONTACT, PATH_MODEL, PATH_MODEL, PATH_MODEL, PATH_MODEL_1_CONTACT)
    phase_time = (0.2, 0.2, 0.3, 0.3, 0.3)
    n_shooting = (20, 20, 30, 30, 30)

    # Solver options
    solver = Solver.IPOPT(show_options=dict(show_bounds=True), _linear_solver="MA57")  # show_online_optim=True,
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_tol(1e-6)

    if WITH_MULTI_START:

        combinatorial_parameters = {
            "bio_model_path": [biorbd_model_path],
            "phase_time": [phase_time],
            "n_shooting": [n_shooting],
            "WITH_MULTI_START": [True],
            "seed": list(range(0, 20)),
        }

        multi_start = prepare_multi_start(
            prepare_ocp,
            save_results=save_results,
            combinatorial_parameters=combinatorial_parameters,
            save_folder=save_folder,
            solver=solver,
            # n_pools=1,
        )

        multi_start.solve()
    else:
        ocp = prepare_ocp(biorbd_model_path[0], phase_time[0], n_shooting[0], WITH_MULTI_START=False)

        sol = ocp.solve(solver)
        sol.print_cost()

        # --- Save results --- #
        sol.graphs(show_bounds=True, save_name=str(movement) + "_V" + version)
        sol.animate()

        combinatorial_parameters = [biorbd_model_path, phase_time, n_shooting, WITH_MULTI_START, "no_seed"]
        save_results(sol, *combinatorial_parameters, save_folder=save_folder)


if __name__ == "__main__":
    main()
