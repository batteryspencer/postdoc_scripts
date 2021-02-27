from pathlib import Path


def compute_ts_energies(ts_data, fs_data, e_f_data, e_ts_ref_data,
                        phi_correction, alk_corr, v_extra):
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
