from bioptim import (
    PenaltyController,
    QuadratureRule,
)


def custom_minimize_q_udot(penalty, controller: PenaltyController):
    """
    Minimize the states variables.
    By default this function is quadratic, meaning that it minimizes towards the target.
    Targets (default=np.zeros()) and indices (default=all_idx) can be specified.

    Parameters
    ----------
    penalty: PenaltyOption
        The actual penalty to declare
    controller: PenaltyController
        The penalty node elements
    """

    penalty.quadratic = True if penalty.quadratic is None else penalty.quadratic
    if (
        penalty.integration_rule != QuadratureRule.APPROXIMATE_TRAPEZOIDAL
        and penalty.integration_rule != QuadratureRule.TRAPEZOIDAL
    ):
        penalty.add_target_to_plot(controller=controller, combine_to="q_udot_states")
    penalty.multi_thread = True if penalty.multi_thread is None else penalty.multi_thread

    # TODO: We should scale the target here!
    return controller.states["q_udot"].cx_start

