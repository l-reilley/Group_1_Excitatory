# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:10:30 2026

@author: laure
"""


import numpy as np
import matplotlib.pyplot as plt

def izhikevich_sim(a, b, c, d, I_fn, T=300.0, dt=0.1, v0=-65.0):
    """
    Simulate the Izhikevich neuron model.
    dv/dt = 0.04v^2 + 5v + 140 - u + I
    du/dt = a(bv - u)
    if v >= 30 mV: v <- c, u <- u + d

    Parameters
    ----------
    a, b, c, d : float
        Izhikevich parameters.
    I_fn : callable
        Function of time t (ms) returning input current.
    T : float
        Total time in ms.
    dt : float
        Time step in ms.
    v0 : float
        Initial membrane potential.
    """
    n = int(T / dt)
    t = np.arange(n) * dt
    v = np.zeros(n)
    u = np.zeros(n)
    I = np.zeros(n)

    v[0] = v0
    u[0] = b * v0

    for i in range(n - 1):
        I[i] = I_fn(t[i])

        dv = 0.04 * v[i]**2 + 5 * v[i] + 140 - u[i] + I[i]
        du = a * (b * v[i] - u[i])

        v[i + 1] = v[i] + dt * dv
        u[i + 1] = u[i] + dt * du

        if v[i + 1] >= 30:
            v[i] = 30  # show spike peak in plot
            v[i + 1] = c
            u[i + 1] += d

    I[-1] = I_fn(t[-1])
    return t, v, u, I


def step_current(start=50, stop=250, amp=10):
    return lambda t: amp if start <= t <= stop else 0.0


if __name__ == "__main__":
    # Regular-spiking excitatory interneuron (RS)
    rs_params = dict(a=0.02, b=0.2, c=-55, d=9.8)

    # Intrinsically bursting excitatory interneuron (IB)
    ib_params = dict(a=0.02, b=0.2, c=-55, d=4)
    
    # Chattering excitatory interneuron (CH)
    ch_params = dict(a=0.02, b=0.2, c=-55, d=1.75)

    I_fn = step_current(start=50, stop=250, amp=10)

    t_rs, v_rs, u_rs, I_rs = izhikevich_sim(**rs_params, I_fn=I_fn)
    t_ib, v_ib, u_ib, I_ib = izhikevich_sim(**ib_params, I_fn=I_fn)
    t_ch, v_ch, u_ch, I_ch = izhikevich_sim(**ch_params, I_fn=I_fn)

    # Create 3 vertical subplots
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
        ax.set_ylabel("Membrane voltage (mV)")

    plt.tight_layout()
    plt.show()
    

    
    
 