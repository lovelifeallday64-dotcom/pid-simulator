
# pid_simulator_streamlit.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO, BytesIO

st.set_page_config(page_title="PID Simulator (mobile-friendly)", layout="centered")

st.title("PID Simulator — web version")
st.markdown("Realistic PID simulation (first/second-order plants + dead time). "
            "Use the sliders to tune, press **Run simulation** to see results. "
            "Help icons (ℹ️) explain each setting.")


# ---------------------------
# Utilities & Models
# ---------------------------
class PID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.1, u_min=None, u_max=None, anti_windup=True, derivative_filter_tau=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.anti_windup = anti_windup
        self.tau = derivative_filter_tau
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_deriv = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_deriv = 0.0

    def update(self, setpoint, pv):
        error = setpoint - pv
        p = self.kp * error
        # trapezoidal integration
        self._integral += 0.5 * (error + self._prev_error) * self.dt
        i = self.ki * self._integral
        raw_deriv = (error - self._prev_error) / self.dt
        alpha = self.dt / (self.tau + self.dt)
        deriv = (1 - alpha) * self._prev_deriv + alpha * raw_deriv
        d = self.kd * deriv

        u = p + i + d

        # clamp
        u_limited = u
        if (self.u_min is not None) and (u < self.u_min):
            u_limited = self.u_min
        if (self.u_max is not None) and (u > self.u_max):
            u_limited = self.u_max

        if self.anti_windup and u != u_limited:
            # undo last integral step (simple anti-windup)
            self._integral -= 0.5 * (error + self._prev_error) * self.dt
            i = self.ki * self._integral
            u = p + i + d

        self._prev_error = error
        self._prev_deriv = deriv

        return u_limited, p, i, d


class FirstOrderPlant:
    def __init__(self, K=1.0, tau=1.0, dt=0.1, y0=0.0):
        self.K = K
        self.tau = tau
        self.dt = dt
        self.y = y0

    def update(self, u):
        dy = (-self.y + self.K * u) / self.tau
        self.y += dy * self.dt
        return self.y


class SecondOrderPlant:
    def __init__(self, K=1.0, wn=1.0, zeta=0.5, dt=0.1, y0=0.0):
        self.K = K
        self.wn = wn
        self.zeta = zeta
        self.dt = dt
        self.y = y0
        self.y_dot = 0.0

    def update(self, u):
        y_dd = self.K * (self.wn**2) * u - 2*self.zeta*self.wn*self.y_dot - (self.wn**2)*self.y
        self.y_dot += y_dd * self.dt
        self.y += self.y_dot * self.dt
        return self.y


# Dead time handler: a circular buffer storing past controller outputs.
class DeadTimeBuffer:
    def __init__(self, delay_seconds=0.0, dt=0.1, initial_value=0.0):
        self.dt = dt
        self.size = max(1, int(np.ceil(delay_seconds / dt)))
        self.buffer = [initial_value] * self.size
        self.idx = 0

    def push_and_get(self, value):
        # push current value into buffer and return delayed output
        delayed_value = self.buffer[self.idx]
        self.buffer[self.idx] = value
        self.idx = (self.idx + 1) % self.size
        return delayed_value


# ---------------------------
# Sidebar: Controls
# ---------------------------
st.sidebar.header("Controller & Simulation")

# Expand slider ranges per your request:
# Original proposed ranges were:
# Kp 0→20 => upper<100 => quadruple => 80
# Ki 0→10 => 40
# Kd 0→10 => 40
# Setpoint 0→100 => upper ==100 => treat as >=100 -> double => 200
# Simulation Time 10→300 => >100 -> double => 600
# Process Gain K 0.1→10 => 40
# Time constant tau 1→120 => >100 -> double => 240
# Damping zeta 0.1→2.0 => 8
# Dead time 0→10 => 40

st.sidebar.subheader("PID Gains")
col_a, col_b = st.sidebar.columns([3,1])
with col_a:
    kp = st.slider("Kp", min_value=0.0, max_value=80.0, value=2.0, step=0.1, help="Proportional gain.")
with col_b:
    with st.expander("ℹ️ Help Kp"):
        st.write("Proportional gain multiplies the current error. Example: If error is 10 and Kp=2, P contribution = 20.")
with col_a:
    ki = st.slider("Ki", min_value=0.0, max_value=40.0, value=1.0, step=0.01, help="Integral gain.")
with col_b:
    with st.expander("ℹ️ Help Ki"):
        st.write("Integral gain accumulates past error, helping eliminate steady-state error. Example: Ki=1 integrates error over time.")
with col_a:
    kd = st.slider("Kd", min_value=0.0, max_value=40.0, value=0.5, step=0.01, help="Derivative gain.")
with col_b:
    with st.expander("ℹ️ Help Kd"):
        st.write("Derivative gain reacts to rate-of-change of error, damping overshoot. Example: Kd=0.5 reduces oscillations.")


st.sidebar.subheader("Setpoint & Sim")
col1, col2 = st.sidebar.columns([2,1])
with col1:
    setpoint = st.slider("Setpoint (SP)", min_value=0.0, max_value=200.0, value=100.0, step=0.1, help="Target process value.")
with col2:
    with st.expander("ℹ️ Help SP"):
        st.write("The target value for the process. Example: For temperature setpoint = 100 (units as you choose).")


sim_time = st.sidebar.slider("Simulation time (s)", min_value=10, max_value=600, value=120, step=5,
                             help="Total simulated seconds.")
dt = st.sidebar.number_input("Sample time / dt (s)", min_value=0.005, max_value=1.0, value=0.05, step=0.005,
                             help="Discrete simulation timestep. Smaller dt = more accurate but slower.")


st.sidebar.subheader("Actuator / limits")
u_min = st.sidebar.number_input("Actuator min (u_min)", value=0.0, step=0.1)
u_max = st.sidebar.number_input("Actuator max (u_max)", value=200.0, step=0.1)


st.sidebar.header("Plant (Process)")
plant_type = st.sidebar.selectbox("Plant type", options=["First Order", "Second Order"], index=0)

colp1, colp2 = st.sidebar.columns([2,1])
with colp1:
    K = st.sidebar.slider("Process gain (K)", min_value=0.1, max_value=40.0, value=1.0, step=0.1,
                          help="Plant gain")
with colp2:
    with st.expander("ℹ️ Help K"):
        st.write("Process gain multiplies control signal to produce process input. Example: gain=2 doubles the effect of controller output.")


tau = st.sidebar.slider("Time constant (τ, s)", min_value=0.1, max_value=240.0, value=30.0, step=0.1,
                        help="How quickly the process responds (higher = slower).")
with st.sidebar.expander("ℹ️ Help τ"):
    st.write("Time constant τ describes how fast the system reacts. Example: τ=60 means ~63% of change reached in 60s for first-order systems.")


zeta = st.sidebar.slider("Damping (ζ) — 2nd order only", min_value=0.1, max_value=8.0, value=0.8, step=0.05,
                         help="Damping ratio for second-order plant.")
with st.sidebar.expander("ℹ️ Help ζ"):
    st.write("Low ζ (e.g., 0.2) => oscillatory; high ζ (e.g., 2.0) => overdamped. Example: ζ=0.5 is underdamped, expect some overshoot.")


dead_time = st.sidebar.slider("Dead time (delay, s)", min_value=0.0, max_value=40.0, value=2.0, step=0.1,
                              help="Transport delay before plant sees control action.")
with st.sidebar.expander("ℹ️ Help dead time"):
    st.write("Dead time simulates delays like heat propagation. Example: dead time=10s means the plant reacts 10s after controller output changes.")


st.sidebar.header("Extras")
noise_std = st.sidebar.slider("Measurement noise std dev", min_value=0.0, max_value=20.0, value=0.0, step=0.1,
                              help="Add gaussian noise to PV (helps test robustness).")
with st.sidebar.expander("ℹ️ Help noise"):
    st.write("Adds random measurement noise to PV. Example: noise_std=0.5 simulates sensor jitter.")


# ---------------------------
# Main area: Run button and plot
# ---------------------------
run_sim = st.button("Run simulation")

# optional quick presets (useful)
st.write("---")
st.markdown("**Quick presets:**")
preset_col1, preset_col2, preset_col3 = st.columns(3)
with preset_col1:
    if st.button("Aggressive"):
        st.session_state.update({'kp': 8.0, 'ki': 10.0, 'kd': 2.0})
with preset_col2:
    if st.button("Moderate"):
        st.session_state.update({'kp': 2.0, 'ki': 1.0, 'kd': 0.5})
with preset_col3:
    if st.button("Conservative"):
        st.session_state.update({'kp': 0.8, 'ki': 0.2, 'kd': 0.05})

# keep UI responsive: store results to display if run
if 'last_df' not in st.session_state:
    st.session_state['last_df'] = None
if 'last_fig' not in st.session_state:
    st.session_state['last_fig'] = None

if run_sim:
    # Build simulation objects
    pid = PID(kp=kp, ki=ki, kd=kd, dt=dt, u_min=u_min, u_max=u_max, anti_windup=True, derivative_filter_tau=max(1e-6, dt/10))
    pid.reset()

    # Plant initialization
    if plant_type == "First Order":
        plant = FirstOrderPlant(K=K, tau=tau, dt=dt, y0=0.0)
    else:
        # choose wn approx from time constant: wn ~ 4/τ for a rough mapping, allow user damping to dominate
        wn = max(0.01, 4.0 / max(0.1, tau))
        plant = SecondOrderPlant(K=K, wn=wn, zeta=zeta, dt=dt, y0=0.0)

    # Dead time buffer
    dt_buffer = DeadTimeBuffer(delay_seconds=dead_time, dt=dt, initial_value=0.0)

    n_steps = int(np.ceil(sim_time / dt)) + 1
    time = np.linspace(0, sim_time, n_steps)
    pv = np.zeros(n_steps)
    u = np.zeros(n_steps)
    sp = np.zeros(n_steps)
    p_term = np.zeros(n_steps)
    i_term = np.zeros(n_steps)
    d_term = np.zeros(n_steps)

    # Simulation loop
    for k in range(n_steps):
        sp[k] = setpoint  # constant setpoint — could be extended to step/ramp profiles
        # controller sees current PV and computes u
        u_k, p_k, i_k, d_k = pid.update(sp[k], plant.y)
        # push u_k into dead-time buffer and get delayed control that the plant actually sees
        u_delayed = dt_buffer.push_and_get(u_k)
        # update plant with delayed control
        pv_k = plant.update(u_delayed)
        # optionally add noise to measured PV
        pv_k_meas = pv_k + np.random.normal(0.0, noise_std) if noise_std > 0 else pv_k

        # Save results
        pv[k] = pv_k_meas
        u[k] = u_k
        p_term[k] = p_k
        i_term[k] = i_k
        d_term[k] = d_k

    # Prepare results dataframe
    df = pd.DataFrame({
        'time_s': time,
        'setpoint': sp,
        'pv': pv,
        'controller_u': u,
        'P': p_term,
        'I': i_term,
        'D': d_term
    })
    st.session_state['last_df'] = df

    # Plot results with matplotlib
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df['time_s'], df['setpoint'], label='Setpoint (SP)')
    ax.plot(df['time_s'], df['pv'], label='Process Variable (PV)')
    ax.plot(df['time_s'], df['controller_u'], label='Controller output (u)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title('PID simulation')
    ax.legend(loc='upper right')
    ax.grid(True)
    st.session_state['last_fig'] = fig

    st.pyplot(fig)

    # CSV download
    csv = df.to_csv(index=False)
    b = BytesIO()
    b.write(csv.encode())
    b.seek(0)
    st.download_button("Download CSV", b, file_name="pid_simulation.csv", mime="text/csv")
else:
    st.info("Adjust sliders in the sidebar and press **Run simulation**. Use the small ℹ️ Help expanders for examples.")

# Show last run (if present)
if st.session_state['last_df'] is not None and not run_sim:
    st.write("### Last simulation (press Run simulation to re-run)")
    st.pyplot(st.session_state['last_fig'])
    st.download_button("Download last CSV", st.session_state['last_df'].to_csv(index=False), file_name="pid_simulation_last.csv", mime="text/csv")

st.write("---")
st.caption("Built for mobile access. For deployment: see instructions below.")

# ---------------------------
# End of file
# ---------------------------
