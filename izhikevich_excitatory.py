# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:10:30 2026

@author: laure
"""


import numpy as np
import matplotlib.pyplot as plt

# Simulate the model
def izhikevich_sim(a, b, c, d, I_fn, T=300.0, dt=0.1, v0=-65.0): # All inputs taken in the function
    
    # Create arrays to store changing variables over time
    n = int(T / dt) # How many time steps
    t = np.arange(n) * dt
    v = np.zeros(n)
    u = np.zeros(n)
    I = np.zeros(n)

    v[0] = v0  # Initialize voltage by calling function input
    u[0] = b * v0  # Intialize membrane recovery variable

    for i in range(n - 1):  # Go through each time step and find change in voltage and membrane recovery variable (dv and du)
        I[i] = I_fn(t[i]) # Input current is current function at time i

        dv = 0.04 * v[i]**2 + 5 * v[i] + 140 - u[i] + I[i]  # Directly from model, change in voltage over time
        du = a * (b * v[i] - u[i])  # Also directly from model, change in membrane recovery variable over time

        # Uses above calculations to calculate next step
        v[i + 1] = v[i] + dt * dv 
        u[i + 1] = u[i] + dt * du
        
        if v[i + 1] >= 30:  # Reset voltage and update variables according to parameters c and d
            v[i] = 30  # show spike peak in plot
            v[i + 1] = c
            u[i + 1] += d

    I[-1] = I_fn(t[-1]) # Input current now at very end of sequence
    return t, v, u, I # Return variables that change in time

# How long the function runs for
def step_current(start=50, stop=250, amp=10):
    return lambda t: amp if start <= t <= stop else 0.0

# Three different spiking patterns for excitatory neurons
if __name__ == "__main__":
    # Regular-spiking excitatory interneuron (RS)
    rs_params = dict(a=0.02, b=0.2, c=-55, d=10)

    # Intrinsically bursting excitatory interneuron (IB)
    ib_params = dict(a=0.02, b=0.2, c=-55, d=4)
    
    # Chattering excitatory interneuron (CH)
    ch_params = dict(a=0.02, b=0.2, c=-55, d=1.75)

    I_fn = lambda t: 10  # Setting current to constant

    # Call the ihikevich model and input parameters we set for each type of behavior
    t_rs, v_rs, u_rs, I_rs = izhikevich_sim(**rs_params, I_fn=I_fn)
    t_ib, v_ib, u_ib, I_ib = izhikevich_sim(**ib_params, I_fn=I_fn)
    t_ch, v_ch, u_ch, I_ch = izhikevich_sim(**ch_params, I_fn=I_fn)

    # Create 3 vertical subplots to plot all types of behavior on one figure
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    fig.suptitle("Izhikevich Excitatory Neuron Dynamics", fontsize=16)
        
    # Plot each neuron
    axs[0].plot(t_rs, v_rs, color='red')
    axs[0].set_title("RS excitatory neuron")

    axs[1].plot(t_ib, v_ib, color='blue')
    axs[1].set_title("IB excitatory neuron")

    axs[2].plot(t_ch, v_ch, color='green')
    axs[2].set_title("CH excitatory neuron")

    axs[2].set_xlabel("Time (ms)")
    for ax in axs:
        ax.set_ylabel("Voltage (mV)")

    plt.tight_layout()
    plt.show()

# Create theta model for chattering neuron
# Get min and max voltage values for CH
v_min, v_max = np.min(v_ch), np.max(v_ch) 

# Create units along pi respective to min and max values
theta_ch = 2*np.pi * (v_ch - v_min) / (v_max - v_min) 
x_ch, y_ch = np.cos(theta_ch), np.sin(theta_ch)

# Create unit circule
plt.figure(figsize=(5,5))
circle = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(circle), np.sin(circle), 'k--') 

# Plot voltage trajectory of chattering neuron on circle
plt.plot(x_ch, y_ch, label='CH (chattering)')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Unit Circle Projection (Chattering neuron)")
plt.show()
 
    

    
    
 
