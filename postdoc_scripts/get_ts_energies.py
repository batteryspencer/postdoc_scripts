from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def compute_ts_energies(input_data, e_f_data, phi_correction, alk_corr,
                        v_extra, rxn_type):
    E_TS_noH = input_data[:, 0]
    q_TS_noH = input_data[:, 1]
    phi_TS_noH = input_data[:, 2]
    phi_TS_noH_corr = phi_TS_noH - phi_correction

    E_FS_noH = input_data[:, 3]
    q_FS_noH = input_data[:, 4]
    phi_FS_noH = input_data[:, 5]
    phi_FS_noH_corr = phi_TS_noH - phi_correction

    E_TS_H = input_data[:, 6]
    q_TS_H = input_data[:, 7]
    phi_TS_H = input_data[:, 8]
    phi_TS_H_corr = phi_TS_noH - phi_correction

    E_FS_H = input_data[:, 9]
    q_FS_H = input_data[:, 10]
    phi_FS_H = input_data[:, 11]
    phi_FS_H_corr = phi_TS_noH - phi_correction

    del_E_noH = E_TS_noH - E_FS_noH
    del_q_noH = q_TS_noH - q_FS_noH
    del_phi_noH = phi_TS_noH_corr - phi_FS_noH_corr
    
    del_E_H = E_TS_H - E_FS_H
    del_q_H = q_TS_H - q_FS_H
    del_phi_H = phi_TS_H_corr - phi_FS_H_corr

    # backward barrier
    E_r_noH = del_E_noH + 0.5 * del_q_noH * del_phi_noH
    E_r_H = del_E_H + 0.5 * del_q_H * del_phi_H
    
    # charge extrapolation. Extrapolate to 4.0 eV (U_RHE = 0 V at pH7).
    E_r_extrapolated_noH = E_r_noH + del_q_noH * (phi_FS_noH_corr - v_extra)
    E_r_extrapolated_H = E_r_H + del_q_H * (phi_FS_H_corr - v_extra)
    
    E_r_alk_noH = E_r_extrapolated_noH + alk_corr * np.asarray([1.0 if element=='Electrochemical' else 0.0 for element in rxn_type])
    E_r_alk_H = E_r_extrapolated_H + alk_corr * np.asarray([1.0 if element=='Electrochemical' else 0.0 for element in rxn_type])
    
    ts_energies_noH = E_r_alk_noH + e_f_data
    ts_energies_H = E_r_alk_H + e_f_data
    return (ts_energies_noH, ts_energies_H)

def plot_ts_energies(ts_states, rxn_type, phi_correction_list, alk_corr,
                     v_extra, input_data_filepath, e_f_data_filepath,
                     ts_ref_data_filepath):

    input_data = np.loadtxt(input_data_filepath)
    e_f_data = np.loadtxt(e_f_data_filepath)
    ts_ref_data = np.loadtxt(ts_ref_data_filepath)

    font_size = 14
    label_size = 12
    tick_size = 10
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ts_state_range = np.arange(len(ts_states))

    num_bars = len(phi_correction_list) * 2
    bar_width = (ts_state_range[1] - ts_state_range[0]) / num_bars / 2

    if num_bars % 2:
        bar_indices = np.arange(-(num_bars // 2), (num_bars // 2) + 1)
    else:
        bar_indices = np.arange(-(num_bars // 2), num_bars // 2)

    index = 0
    for phi_correction in phi_correction_list:
        (ts_energies_noH, ts_energies_H) = compute_ts_energies(
            input_data, e_f_data, phi_correction, alk_corr, v_extra, rxn_type)
        
        # deviation
        diff_noH = ts_energies_noH - ts_ref_data
        diff_H = ts_energies_H - ts_ref_data

        if num_bars % 2:
            ax.bar(ts_state_range + bar_indices[index] * bar_width, diff_noH, width=bar_width, align='center', label=f'wf_corr={phi_correction:.1f}, No H in $T_{{Ads}}$')
            index += 1
            ax.bar(ts_state_range + bar_indices[index] * bar_width, diff_H, width=bar_width, align='center', label=f'wf_corr={phi_correction:.1f}, H in $T_{{Ads}}$')
        else:
            ax.bar(ts_state_range + bar_indices[index] * bar_width, diff_noH, width=bar_width, align='edge', label=f'wf_corr={phi_correction:.1f}, No H in $T_{{Ads}}$')
            index += 1
            ax.bar(ts_state_range + bar_indices[index] * bar_width, diff_H, width=bar_width, align='edge', label=f'wf_corr={phi_correction:.1f}, H in $T_{{Ads}}$')
        index += 1

    ax.legend()
    ax.set_xlabel('Transition State', fontsize=font_size)
    ax.set_ylabel('Energy (eV)', fontsize=font_size)
    ax.set_title(f'Validation Errors for TS at epsilon=0.0')
    ax.set_xticks(ts_state_range)
    ax.set_xticklabels(ts_states, rotation=90, fontsize=tick_size)
    yticks = plt.yticks()[0]
    plt.tight_layout()
    figure_name = f'Validation of TS Calculations.png'
    figure_path = input_data_filepath.parent / figure_name
    plt.savefig(figure_path, dpi=600)
    return None
