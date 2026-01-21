# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 20:50:28 2026

@author: lwagner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# Load data
DF_GFP = pd.read_csv("GFP_data.csv")

# ODE: production minus degradation
def mg_ode(M, t, k_prod, k_deg):
    return k_prod - k_deg * M

# Solve ODE for curve_fit
def mg_ode_solution(t, k_prod, k_deg, M0):
    M = odeint(mg_ode, M0, t, args=(k_prod, k_deg))
    return M.flatten()

# Fit + plot function
def fit_and_plot_buffer_ode_peak(DF, buffer_name):
    time = DF["time"].values
    MG = DF[buffer_name].values

    # Initial guess: production rate, degradation rate, initial value
    p0 = [5000, 0.05, MG[0]]  

    # Fit ODE model
    popt, pcov = curve_fit(mg_ode_solution, time, MG, p0=p0, bounds=(0, np.inf))
    k_prod_fit, k_deg_fit, M0_fit = popt

    # Smooth prediction
    time_fit = np.linspace(time.min(), time.max(), 300)
    MG_fit = mg_ode_solution(time_fit, k_prod_fit, k_deg_fit, M0_fit)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(time, MG, label="Experimental data")
    plt.plot(time_fit, MG_fit, label=f"ODE fit\nk_prod={k_prod_fit:.1f}, k_deg={k_deg_fit:.3f}")
    plt.xlabel("Time (min)")
    plt.ylabel("MG fluorescence")
    plt.title(buffer_name)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Execution
for buffer_name in DF_GFP.columns[1:]:  # skip 'time'
    fit_and_plot_buffer_ode_peak(DF_GFP, buffer_name)
