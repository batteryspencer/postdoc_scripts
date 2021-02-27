from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def compute_ts_energies(ts_data, fs_data, e_f_data, phi_correction, alk_corr,
                        v_extra, rxn_type):
    E_TS = ts_data[:, 0]
    q_TS = ts_data[:, 1]
    phi_TS = ts_data[:, 2]
    phi_TS_corr = phi_TS - phi_correction
    
    E_FS = fs_data[:, 0]
    q_FS = fs_data[:, 1]
    phi_FS = fs_data[:, 2]
    phi_FS_corr = phi_FS - phi_correction
    
    del_E = E_TS - E_FS
    del_q = q_TS - q_FS
    del_phi = phi_TS_corr - phi_FS_corr
    
    # backward barrier
    E_r = del_E + 0.5 * del_q * del_phi
    
    # charge extrapolation. Extrapolate to 4.0 eV (U_RHE = 0 V at pH7).
    E_r_extrapolated = E_r + del_q * (phi_FS_corr - v_extra)
    
    E_r_alk = E_r_extrapolated + alk_corr * np.asarray([1.0 if element=='Electrochemical' else 0.0 for element in rxn_type])
    
    ts_energies = E_r_alk + e_f_data
    return ts_energies

def plot_ts_energies(ts_states, rxn_type, phi_correction_list, alk_corr,
                     v_extra, ts_data_filepath, fs_data_filepath,
                     e_f_data_filepath, ts_ref_data_filepath, subtitle, suffix):

    ts_data = np.loadtxt(ts_data_filepath)
    fs_data = np.loadtxt(fs_data_filepath)
    e_f_data = np.loadtxt(e_f_data_filepath)
    ts_ref_data = np.loadtxt(ts_ref_data_filepath)

    font_size = 14
    label_size = 12
    tick_size = 10
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ts_state_range = np.arange(len(ts_states))

    num_bars = len(phi_correction_list)
    bar_width = (ts_state_range[1] - ts_state_range[0]) / num_bars / 2
    
    bar_index = 0
    for phi_correction in phi_correction_list:
        bar_index += 1
        ts_energies = compute_ts_energies(ts_data, fs_data, e_f_data,
                                          phi_correction, alk_corr, v_extra,
                                          rxn_type)
        
        # deviation
        diff = ts_energies - ts_ref_data

        ax.bar(ts_state_range + bar_index * bar_width, diff, width=bar_width, align='center', label=f'wf_corr={phi_correction:.1f}')

    ax.legend()
    ax.set_xlabel('Transition State', fontsize=font_size)
    ax.set_ylabel('Energy (eV)', fontsize=font_size)
    ax.set_title(f'Validation Errors for TS at epsilon=0.0\n{subtitle}')
    ax.set_xticks(ts_state_range)
    ax.set_xticklabels(ts_states, rotation=90, fontsize=tick_size)
    yticks = plt.yticks()[0]
    plt.tight_layout()
    figure_name = f'Validation of TS Calculations_{suffix}.png'
    plt.savefig(figure_name, dpi=600)
    return None
