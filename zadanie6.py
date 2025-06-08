import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass


@dataclass
class Data:
    J1: float
    J2: float
    b1: float
    b2: float
    n1: float
    n2: float
    f: float
    A: float
    dt: float
    Tsym:float
    n:float


def stan(t,u, x,dataset):

    Jeq = dataset.J1 + dataset.J2 * (dataset.n1 / dataset.n2)**2
    beq = dataset.b1 + dataset.b2 * (dataset.n1 / dataset.n2)**2

    dtheta1 = x[1]
    domega1 = (u(t) - beq * x[1]) / Jeq

    return np.array([dtheta1, domega1])

def tm(t,dataset):
    if dataset.n==1:
        u = dataset.A*signal.sawtooth(2 * np.pi * dataset.f * t, width=0.5)
    elif dataset.n==2:
        u = dataset.A*np.sin(2 * np.pi * dataset.f * t)
    elif dataset.n==3:
        u = np.zeros_like(t)
        u[(t < 0.5 * dataset.Tsym)] = dataset.A
    return u

def Euler(x0,u,t,dataset):
    x_euler = np.zeros((len(t), 2))
    x_euler[0] = x0

    for i in range(1, len(t)):
        dx = stan(t[i-1], u, x_euler[i-1],dataset)
        x_euler[i] = x_euler[i-1] + dataset.dt * dx

    return x_euler

def RK4(x0, u, t,dataset):
    x_rk = np.zeros((len(t), 2))
    x_rk[0] = x0

    for i in range(1, len(t)):
        ti = t[i-1]
        xi = x_rk[i-1]

        k1 = stan(ti, u, xi,dataset)
        k2 = stan(ti + dataset.dt/2, u, xi + dataset.dt/2 * k1,dataset)
        k3 = stan(ti + dataset.dt/2, u, xi + dataset.dt/2 * k2,dataset)
        k4 = stan(ti + dataset.dt, u, xi + dataset.dt * k3,dataset)

        x_rk[i] = xi + (dataset.dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return x_rk



def run_sim():
 
   
    x0 = [float(entries["θ1(0)"].get()), float(entries["ω1(0)"].get())]
    

    dataset = Data(
        J1 = float(entries["J1"].get()),
        J2 = float(entries["J2"].get()),
        b1 = float(entries["b1"].get()),
        b2 = float(entries["b2"].get()),
        n1 = float(entries["n1"].get()),
        n2 = float(entries["n2"].get()),
        dt = float(entries["dt"].get()),
        Tsym = float(entries["Tsym"].get()),
        f = float(entries["f"].get()),
        A = float(entries["A"].get()),
        n = signal_type.get()
        )

    t = np.arange(0, dataset.Tsym, dataset.dt)
    u = tm(t,dataset)
    u_func = interp1d(t, u, fill_value="extrapolate")

    x_euler = Euler(x0, u_func, t, dataset)
    theta2e = x_euler[:, 0] * (dataset.n1 / dataset.n2)
    omega2e = x_euler[:, 1] * (dataset.n1 / dataset.n2)

    x_rk = RK4(x0, u_func, t,  dataset)
    theta2rk = x_rk[:, 0] * (dataset.n1 / dataset.n2)
    omega2rk = x_rk[:, 1] * (dataset.n1 / dataset.n2)

    plt.figure(figsize=(10, 5))
    plt.plot(t, u)
    plt.grid()
    plt.title("Sygnał wejściowy")
    plt.xlabel("Czas [s]")
    

    plt.figure(figsize=(10, 5))
    plt.plot(t, theta2e, label='θ2 (Euler)')
    plt.plot(t, omega2e, label='ω2 (Euler)')
    plt.plot(t, theta2rk, label='θ2 (RK4)', linestyle='--')
    plt.plot(t, omega2rk, label='ω2 (RK4)', linestyle='--')
    plt.grid()
    plt.legend()
    plt.title("Wynik symulacji")
    plt.xlabel("Czas [s]")
    plt.show()


root = tk.Tk()
root.title("Symulator układu – RK4 / Euler")

frame = tk.Frame(root)
frame.pack(padx=100, pady=10)

param_names = ["J1", "J2", "b1", "b2", "n1", "n2", "dt", "Tsym", "θ1(0)", "ω1(0)", "f", "A"]
entries = {}

for i, name in enumerate(param_names):
    tk.Label(frame, text=name).grid(row=i, column=0, sticky="w")
    entry = tk.Entry(frame)
    entry.grid(row=i, column=1)
    entries[name] = entry


defaults = {
    "J1": 0.01, "J2": 0.02, "b1": 0.05, "b2": 0.1, "n1": 1, "n2": 2,
    "dt": 0.01, "Tsym": 10, "θ1(0)": 0, "ω1(0)": 0,
    "f": 1, "A": 1
}
for key in defaults:
    entries[key].insert(0, str(defaults[key]))


signal_type = tk.IntVar(value=2)
tk.Label(frame, text="Rodzaj sygnału:").grid(row=0, column=2, sticky="w")
signals = [ ("Trójkątny", 1), ("Sinusoidalny", 2), ("Impuls prostokątny", 3)]
for i, (label, val) in enumerate(signals):
    ttk.Radiobutton(frame, text=label, variable=signal_type, value=val).grid(row=i+1, column=2, sticky="w")

tk.Button(root, text="Start symulacji", command=run_sim).pack(pady=10)

root.mainloop()