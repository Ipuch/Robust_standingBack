import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../holonomic_research/")
from plot_actuators import actuator_function, Joint

# --- Save --- #
def get_created_data_from_pickle(file: str):
    """
    This code is used to open a pickle document and exploit its data.

    Parameters
    ----------
    file: path of the pickle document

    Returns
    -------
    data: All the data of the pickle document
    """
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break

    return data_tmp

def time_to_percentage(time):
    time_pourcentage = np.zeros(time.shape)
    for i in range(0, time.shape[0]):
       time_pourcentage[i] = time[i] * 100 / time[time.shape[0]-1]      
    return time_pourcentage

# --- Graph --- #
def graph_all_comparaison(sol_holo, sol2):
    # Sol 1
    # data = get_created_data_from_pickle(sol_holo)
    data = pd.read_pickle(sol_holo)
    lambdas = data["lambda"]
    q_rad = data["q_all"]
    q_rad[6, :] = q_rad[6, :] * -1
    q_rad_without_last_node = np.hstack((q_rad[:, :20], q_rad[:, 21:21+20], q_rad[:, 21+21:21+21+30], q_rad[:, 21+21+31:21+21+31+30], q_rad[:, 21+21+31+31:21+21+31+31+30]))
    q_deg = np.vstack([q_rad[0:2,:], q_rad[2:, :] *180/np.pi]) 
    qdot_rad = data["qdot_all"]
    qdot_rad[6, :] = qdot_rad[6, :] * -1
    qdot_deg = np.vstack([qdot_rad[0:2, :], qdot_rad[2:, :] * 180 / np.pi])
    # qdot_deg = qdot_rad
    # qdot_deg = qdot_rad*180/np.pi
    qddot = data["qddot_all"]
    tau_deg = data["tau_all"]
    dof_names = ["Pelvis \n(Translation Y)", "Pelvis \n(Translation Z)",
                 "Pelvis", "Shoulder", "Elbow",
                 "Hip", "Knee", "Ankle"]
    dof_names_tau = ["Shoulder", "Elbow",
                     "Hip", "Knee", "Ankle"]
    time_total = data["time"][-1][-1]

    time_end_phase = []
    time_end_phase_tau = []
    for i in range(len(data["time"])):
        time_end_phase.append(data["time"][i][-1])
        time_end_phase_tau.append(data["time"][i][-2])
    time_end_phase_pourcentage = time_to_percentage(np.vstack(time_end_phase))
    time_end_phase_tau_pourcentage = time_to_percentage(np.vstack(time_end_phase_tau))

    time = np.vstack(data["time"])
    time_pourcentage = time_to_percentage(time)
    timetau = np.vstack([arr[:-1, :] for arr in data["time"]])
    time_tau_pourcentage = time_to_percentage(timetau)
    # min_bounds_q = data["min_bounds"]
    # max_bounds_q = data["max_bounds"]

    # Sol 2
    data2 = get_created_data_from_pickle(sol2)
    q2_rad = data2["q_all"]
    q2_rad[6, :] = q2_rad[6, :] * -1
    q2_rad_without_last_node = np.hstack((q2_rad[:, :20], q2_rad[:, 21:21+20], q2_rad[:, 21+21:21+21+30], q2_rad[:, 21+21+31:21+21+31+30], q2_rad[:, 21+21+31+31:21+21+31+31+30]))
    q2_deg = np.vstack([q2_rad[0:2,:], q2_rad[2:, :] *180/np.pi])
    qdot2_rad = data2["qdot_all"]
    qdot2_rad[6, :] = qdot2_rad[6, :] * -1
    # qdot2_deg = qdot2_rad
    qdot2_deg = np.vstack([qdot2_rad[0:2, :], qdot2_rad[2:, :] * 180 / np.pi])
    tau2_deg = data2["tau_all"]
    time_total2 = data2["time"][-1][-1]

    time_end_phase2 = []
    time_end_phase_tau2 = []
    for i in range(len(data2["time"])):
        time_end_phase2.append(data2["time"][i][-1])
        time_end_phase_tau2.append(data2["time"][i][-2])
    time_end_phase2_pourcentage = time_to_percentage(np.vstack(time_end_phase2))
    time_end_phase_tau2_pourcentage = time_to_percentage(np.vstack(time_end_phase_tau2))

    time2 = np.vstack(data2["time"])  # data2["time_all"]
    time_pourcentage2 = time_to_percentage(time2)
    time2tau = np.vstack([arr[:-1, :] for arr in data2["time"]])

    # Figure q
    fig, axs = plt.subplots(3, 3, figsize=(10, 5))
    num_col = 0
    num_line = 0
    y_max_1 = np.max([abs(q_deg[0:2, :]), abs(q2_deg[0:2, :])])
    y_max_2 = np.max([abs(q_deg[2:5, :]), abs(q2_deg[2:5, :])])
    y_max_3 = np.max([abs(q_deg[5:, :]), abs(q2_deg[5:, :])])
    # y_min = np.min([q_deg, q2_deg])
    # y_max = np.max([q_deg, q2_deg])
    for nb_seg in range(q_deg.shape[0]):
        axs[num_line, num_col].plot(time_pourcentage2, q2_deg[nb_seg], color="tab:blue", label="without \nconstraints",
                                    alpha=0.75, linewidth=1)
        axs[num_line, num_col].plot(time_pourcentage, q_deg[nb_seg], color="tab:orange",
                                    label="with holonomics \nconstraints", alpha=0.75, linewidth=1)
        for xline in range(len(time_end_phase)):
            axs[num_line, num_col].axvline(time_end_phase2_pourcentage[xline], color="tab:blue", linestyle="--",
                                           linewidth=0.7)
            axs[num_line, num_col].axvline(time_end_phase_pourcentage[xline], color="tab:orange", linestyle="--",
                                           linewidth=0.7)
            # if xline!=2:
            # for node in range(0,3):
            # if xline==0:
            #    axs[num_line, num_col].axhline(y=max_bounds_q[xline][nb_seg,1],
            #                                   xmin=0,
            #                                   xmax=float(time_end_phase_pourcentage[0])/100, color="k",
            #                                   linestyle="-", linewidth=0.7)
            #    axs[num_line, num_col].axhline(y=min_bounds_q[xline][nb_seg,1],
            #                                   xmin=0,
            #                                   xmax=float(time_end_phase_pourcentage[0])/100, color="k",
            #                                   linestyle="-", linewidth=0.7)
            # else:
            #    axs[num_line, num_col].axhline(y=max_bounds_q[xline][nb_seg,1],
            #                               xmin=float(time_end_phase_pourcentage[xline-1])/100,
            #                               xmax=float(time_end_phase_pourcentage[xline])/100, color="k",
            #                               linestyle="-", linewidth=0.7)
            #    axs[num_line, num_col].axhline(y=min_bounds_q[xline][nb_seg,1],
            #                               xmin=float(time_end_phase_pourcentage[xline-1])/100,
            #                               xmax=float(time_end_phase_pourcentage[xline])/100, color="k",
            #                               linestyle="-", linewidth=0.7)
        axs[num_line, num_col].set_title(dof_names[nb_seg], fontsize=8)
        # axs[num_line, num_col].set_ylim(y_min, y_max)
        axs[num_line, num_col].set_xlim(0, 100)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis='both', which='major', labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        elif num_line == 1:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))
        elif num_line == 2:
            axs[num_line, num_col].set_ylim(-y_max_3 + (-y_max_3 * 0.1), y_max_3 + (y_max_3 * 0.1))

        num_col = num_col + 1
        if nb_seg == 1:
            num_col = 0
            num_line += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 2:
            axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)

        # Y_label
        axs[0, 0].set_ylabel("Position [m]", fontsize=7)  # Pelvis Translation
        # axs[0, 1].set_ylabel("Position [m]", fontsize=8) # Pelvis Translation
        axs[1, 0].set_ylabel("F (+) / E (-) [deg]", fontsize=7)  # Pelvis Rotation
        # axs[1, 1].set_ylabel("F (+) / E (-) [rad]", fontsize=8) # Arm Rotation
        # axs[1, 2].set_ylabel("F (+) / E (-) [rad]", fontsize=8) # Forearm Rotation
        axs[2, 0].set_ylabel("F (+) / E (-) [deg]", fontsize=7)  # Thight Rotation
        # axs[2, 1].set_ylabel("F (-) / E (+) [rad]", fontsize=8) # Leg Rotation
        # axs[2, 2].set_ylabel("F (+) / E (-) [rad]", fontsize=8) # Foot Rotation
        # Récupérer les handles et labels de la légende de la figure de la première ligne, première colonne
        handles, labels = axs[0, 0].get_legend_handles_labels()

        # Ajouter la légende à la figure de la première ligne, troisième colonne
        axs[0, 2].legend(handles, labels, loc='center', fontsize=8)
        axs[0, 2].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=1)
    fig.savefig("q.png", format="png")

    # Figure qdot
    fig, axs = plt.subplots(3, 3, figsize=(10, 5))
    num_col = 0
    num_line = 0
    y_max_1 = np.max([abs(qdot_deg[0:2, :]), abs(qdot_deg[0:2, :])])
    y_max_2 = np.max([abs(qdot_deg[2:5, :]), abs(qdot_deg[2:5, :])])
    y_max_3 = np.max([abs(qdot_deg[5:, :]), abs(qdot_deg[5:, :])])
    # y_min = np.min([qdot_deg, qdot2_deg])
    # y_max = np.max([qdot_deg, qdot2_deg])
    for nb_seg in range(qdot_deg.shape[0]):
        axs[num_line, num_col].plot(time_pourcentage2, qdot2_deg[nb_seg], color="tab:blue",
                                    label="without \nconstraints", alpha=0.75, linewidth=1)
        axs[num_line, num_col].plot(time_pourcentage, qdot_deg[nb_seg], color="tab:orange",
                                    label="with holonomics \nconstraints", alpha=0.75, linewidth=1)
        for xline in range(len(time_end_phase)):
            axs[num_line, num_col].axvline(time_end_phase2_pourcentage[xline], color="tab:blue", linestyle="--",
                                           linewidth=0.7)
            axs[num_line, num_col].axvline(time_end_phase_pourcentage[xline], color="tab:orange", linestyle="--",
                                           linewidth=0.7)
        axs[num_line, num_col].set_title(dof_names[nb_seg], fontsize=8)

        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        elif num_line == 1:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))
        elif num_line == 2:
            axs[num_line, num_col].set_ylim(-y_max_3 + (-y_max_3 * 0.1), y_max_3 + (y_max_3 * 0.1))
        # axs[num_line, num_col].set_ylim(y_min, y_max)
        axs[num_line, num_col].set_xlim(0, 100)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis='both', which='major', labelsize=6)

        num_col = num_col + 1
        if nb_seg == 1:
            num_col = 0
            num_line += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 2:
            axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)

        # Y_label
        axs[0, 0].set_ylabel("Velocity [m/s]", fontsize=7)  # Pelvis Translation
        axs[0, 1].set_yticklabels([])  # Pelvis Translation
        axs[1, 0].set_ylabel("F (+) / E (-) [r$\^circ$/s]", fontsize=7)  # Pelvis Rotation
        axs[1, 1].set_yticklabels([])  # Arm Rotation
        axs[1, 2].set_yticklabels([])  # Forearm Rotation
        axs[2, 0].set_ylabel("F (+) / E (-) [r$\^circ$/s]", fontsize=7)  # Thight Rotation
        axs[2, 1].set_yticklabels([])  # Leg Rotation
        axs[2, 2].set_yticklabels([])  # Foot Rotation
        # Récupérer les handles et labels de la légende de la figure de la première ligne, première colonne
        handles, labels = axs[0, 0].get_legend_handles_labels()

        # Ajouter la légende à la figure de la première ligne, troisième colonne
        axs[0, 2].legend(handles, labels, loc='center', fontsize=7)
        axs[0, 2].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    fig.savefig("qdot.png", format="png")


    # Theoretical min and max bound on tau based on actuators
    actuators = {"Shoulders": Joint(tau_max_plus=112.8107*2,
                                    theta_opt_plus=-41.0307*np.pi/180,
                                    r_plus=109.6679*np.pi/180,
                                    tau_max_minus=162.7655*2,
                                    theta_opt_minus=-101.6627*np.pi/180,
                                    r_minus=103.9095*np.pi/180,
                                    min_q=-0.7,
                                    max_q=3.1),
                "Elbows": Joint(tau_max_plus=80*2,
                                theta_opt_plus=np.pi/2-0.1,
                                r_plus=40*np.pi/180,
                                tau_max_minus=50*2,
                                theta_opt_minus=np.pi/2-0.1,
                                r_minus=70*np.pi/180,
                                min_q=0,
                                max_q=2.09),  # this one was not measured, I just tried to fit https://www.researchgate.net/figure/Maximal-isometric-torque-angle-relationship-for-elbow-extensors-fitted-by-polynomial_fig3_286214602
                "Hips": Joint(tau_max_plus=220.3831*2,
                            theta_opt_plus=25.6939*np.pi/180,
                            r_plus=56.4021*np.pi/180,
                            tau_max_minus=490.5938*2,
                            theta_opt_minus=72.5836*np.pi/180,
                            r_minus=48.6999*np.pi/180,
                            min_q=-0.4,
                            max_q=2.6),
                "Knees": Joint(tau_max_plus=367.6643*2,
                            theta_opt_plus=-61.7303*np.pi/180,
                            r_plus=31.7218*np.pi/180,
                            tau_max_minus=177.9694*2,
                            theta_opt_minus=-33.2908*np.pi/180,
                            r_minus=57.0370*np.pi/180,
                            min_q=-2.3,
                            max_q=0.02),
                "Ankles": Joint(tau_max_plus=153.8230*2,
                            theta_opt_plus=0.7442*np.pi/180,
                            r_plus=58.9832*np.pi/180,
                            tau_max_minus=171.9903*2,
                            theta_opt_minus=12.6824*np.pi/180,
                            r_minus=21.8717*np.pi/180,
                            min_q=-0.7,
                            max_q=0.7)
            }

    tau_min_bound = np.zeros((5, tau_deg.shape[1]))
    tau_max_bound = np.zeros((5, tau_deg.shape[1]))
    tau2_min_bound = np.zeros((5, tau2_deg.shape[1]))
    tau2_max_bound = np.zeros((5, tau2_deg.shape[1]))
    for nb_seg, key in enumerate(actuators.keys()):
        tau_min_bound[nb_seg, :] = -actuator_function(actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q_rad_without_last_node[nb_seg+3])
        tau_max_bound[nb_seg, :] = actuator_function(actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q_rad_without_last_node[nb_seg+3])
        tau2_min_bound[nb_seg, :] = -actuator_function(actuators[key].tau_max_minus, actuators[key].theta_opt_minus, actuators[key].r_minus, q2_rad_without_last_node[nb_seg+3])
        tau2_max_bound[nb_seg, :] = actuator_function(actuators[key].tau_max_plus, actuators[key].theta_opt_plus, actuators[key].r_plus, q2_rad_without_last_node[nb_seg+3])

    # Figure tau
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    num_col = 1 
    num_line = 0

    y_max_1 = np.max([abs(tau2_deg[0:2, :]), abs(tau_deg[0:2, :])])
    y_max_2 = np.max([abs(tau2_deg[2:, :]), abs(tau_deg[2:, :])])

    axs[0, 0].plot([], [], color="tab:orange", label="with holonomics \nconstraints")
    axs[0, 0].plot([], [], color="tab:blue", label="without \nconstraints")
    axs[0, 0].legend(loc='center right', bbox_to_anchor=(0.6, 0.5), fontsize=8)
    axs[0, 0].axis('off')

    for nb_seg in range(tau_deg.shape[0]):
        axs[num_line, num_col].step(range(len(tau2_deg[nb_seg])), tau2_max_bound[nb_seg], color="tab:blue", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].step(range(len(tau2_deg[nb_seg])), tau2_min_bound[nb_seg], color="tab:blue", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].step(range(len(tau_deg[nb_seg])), tau_max_bound[nb_seg], color="tab:orange", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].step(range(len(tau_deg[nb_seg])), tau_min_bound[nb_seg], color="tab:orange", alpha=0.5, linewidth=0.5)
        axs[num_line, num_col].step(range(len(tau2_deg[nb_seg])), tau2_deg[nb_seg], color="tab:blue", alpha=0.75, linewidth=1, label="without \nconstraints", where='mid')
        axs[num_line, num_col].step(range(len(tau_deg[nb_seg])), tau_deg[nb_seg], color="tab:orange", alpha=0.75, linewidth=1, label="with holonomics \nconstraints", where='mid')

        for xline in range(len(time_end_phase)):
            axs[num_line, num_col].axvline(time_end_phase_pourcentage[xline], color="tab:orange", linestyle="--",
                                           linewidth=0.7)
            axs[num_line, num_col].axvline(time_end_phase2_pourcentage[xline], color="tab:blue", linestyle="--",
                                           linewidth=0.7)
        axs[num_line, num_col].set_title(dof_names_tau[nb_seg], fontsize=8)
        axs[num_line, num_col].set_xlim(0, 100)
        axs[num_line, num_col].grid(True, linewidth=0.4)
        # Réduire la taille des labels des xticks et yticks
        axs[num_line, num_col].tick_params(axis='both', which='major', labelsize=6)
        if num_line == 0:
            axs[num_line, num_col].set_ylim(-y_max_1 + (-y_max_1 * 0.1), y_max_1 + (y_max_1 * 0.1))
        else:
            axs[num_line, num_col].set_ylim(-y_max_2 + (-y_max_2 * 0.1), y_max_2 + (y_max_2 * 0.1))

        num_col += 1
        if num_col == 3:
            num_col = 0
            num_line += 1
        if num_line == 1:
            axs[num_line, num_col].set_xlabel('Time [%]', fontsize=7)

    # Y_label
    axs[0, 1].set_ylabel("Joint torque [Nm]", fontsize=7)  # Arm Rotation
    axs[1, 0].set_ylabel("Joint torque [Nm]", fontsize=7)  # Leg Rotation
    axs[0, 2].set_yticklabels([])
    axs[1, 1].set_yticklabels([])
    axs[1, 2].set_yticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig("tau.png", format="png")

    # Figure lambdas
    time_tuck = data["time"][2] - data["time"][2][0]
    fig = plt.figure()
    plt.plot(time_tuck, lambdas[0], color='r', label=["Normal force"])
    plt.plot(time_tuck, lambdas[1], color='g', label=["Shear force"])
    plt.ylabel("Force on the tibia [N]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.show()
    fig.savefig("lambdas.png", format="png")
