import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Simple Live PID", layout="centered")
st.title("⚙️ Simple Live PID Simulator")

kp = st.slider("Kp", 0.0, 10.0, 2.0, 0.1)
ki = st.slider("Ki", 0.0, 5.0, 1.0, 0.1)
kd = st.slider("Kd", 0.0, 5.0, 0.5, 0.1)
setpoint = st.slider("Setpoint", 0.0, 100.0, 50.0, 1.0)
run_time = st.slider("Duration (seconds)", 1, 30, 10)
st.write("Tap **Run** to watch how PID responds live.")

if st.button("Run"):
    dt = 0.05
    pv = 0
    integral = 0
    prev_error = 0

    chart = st.empty()
    times, values = [], []

    for t in np.arange(0, run_time, dt):
        error = setpoint - pv
        integral += error * dt
        derivative = (error - prev_error) / dt
        prev_error = error
        u = kp*error + ki*integral + kd*derivative
        pv += (u - pv) * 0.1  # simple process

        times.append(t)
        values.append(pv)

        fig, ax = plt.subplots()
        ax.plot(times, values, label="PV")
        ax.plot(times, [setpoint]*len(times), "--", label="Setpoint")
        ax.set_xlim(0, run_time)
        ax.set_ylim(0, 1.2*max(setpoint, 1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        chart.pyplot(fig)
        time.sleep(dt)
