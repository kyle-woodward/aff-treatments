#%%
import numpy as np
# Generalized point sampling overshoot multipliers used by the ee_treatments() function in main notebook, 
# we adjust these multipliers to oversample certain number of points so that ee_treatments() can 
# generate the treatments with required spacing while ensuring that the spacing filters and masking routines do not 
# leave us with fewer treatment points than are required by the prescription

# think of the float number keys as bins
_dict = {"log":
         {"0.00": {"sm_overshoot":0, "med_overshoot":0, "default_overshoot":0, "mask_spacing": 0, "pt_spacing": 0},
         "0.05": {"sm_overshoot":2.0, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.10": {"sm_overshoot":2.0, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.15": {"sm_overshoot":2.0, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.20": {"sm_overshoot":2.0, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.25": {"sm_overshoot":2.5, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.30": {"sm_overshoot":3.0, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.35": {"sm_overshoot":3.0, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.40": {"sm_overshoot":3.0, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 1.0, "pt_spacing": 1.9},
         "0.45": {"sm_overshoot":3.0, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 1.0, "pt_spacing": 1.9},
         "0.50": {"sm_overshoot":3.0, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 1.0, "pt_spacing": 1.9},
         "0.55": {"sm_overshoot":3.5, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 0.8, "pt_spacing": 1.7},
         "0.60": {"sm_overshoot":3.75, "med_overshoot":2.5, "default_overshoot":1.5, "mask_spacing": 0.7, "pt_spacing": 1.6}
         },
         
         "norm":
          {"0.00": {"sm_overshoot":0, "med_overshoot":0, "default_overshoot":0, "mask_spacing": 0, "pt_spacing": 0},
         "0.05": {"sm_overshoot":2.0, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.10": {"sm_overshoot":2.25, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.15": {"sm_overshoot":2.25, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.20": {"sm_overshoot":2.50, "med_overshoot":1.5, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.25": {"sm_overshoot":2.75, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.30": {"sm_overshoot":3.25, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.35": {"sm_overshoot":3.50, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 1.1, "pt_spacing": 2.01},
         "0.40": {"sm_overshoot":3.75, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 1.0, "pt_spacing": 1.9},
         "0.45": {"sm_overshoot":3.75, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 1.0, "pt_spacing": 1.9},
         "0.50": {"sm_overshoot":4.0, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 1.0, "pt_spacing": 1.9},
         "0.55": {"sm_overshoot":4.25, "med_overshoot":2.0, "default_overshoot":1.5, "mask_spacing": 0.8, "pt_spacing": 1.7},
         "0.60": {"sm_overshoot":4.75, "med_overshoot":2.5, "default_overshoot":1.5, "mask_spacing": 0.7, "pt_spacing": 1.6}
         }
        }

def setall(d, keys, value):
    for k in keys:
        d[k] = value


def get_dials(pct_trt:float,distro:str):
    _dict_distro = _dict[distro]
    _dict_distro_all_keys = {}
    # bin all possible floats within defined range to the generalized binned keys in _dict
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.0, 0.05, 0.001)], _dict_distro["0.05"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.05, 0.1, 0.001)], _dict_distro["0.10"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.1, 0.15, 0.001)], _dict_distro["0.15"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.15, 0.2, 0.001)], _dict_distro["0.20"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.2, 0.25, 0.001)], _dict_distro["0.25"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.25, 0.3, 0.001)], _dict_distro["0.30"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.3, 0.35, 0.001)], _dict_distro["0.35"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.35, 0.4, 0.001)], _dict_distro["0.40"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.4, 0.45, 0.001)], _dict_distro["0.45"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.45, 0.5, 0.001)], _dict_distro["0.50"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.5, 0.55, 0.001)], _dict_distro["0.55"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.55, 0.6, 0.001)], _dict_distro["0.60"])
    setall(_dict_distro_all_keys, [round(i,3) for i in np.arange(0.6, 0.65, 0.001)], _dict_distro["0.60"])
    if pct_trt not in _dict_distro_all_keys.keys():
        raise ValueError("percent treatment value {pct_trt} is not in all_keys dictionary")
    indexed_dict = _dict_distro_all_keys[pct_trt]
    return indexed_dict.values()

#%%
# test
# sm_overshoot,med_overshoot,default_overshoot, mask_spacing, pt_spacing = get_dials(0.61,"norm")
# print(sm_overshoot,med_overshoot,default_overshoot, mask_spacing, pt_spacing)
# %%
