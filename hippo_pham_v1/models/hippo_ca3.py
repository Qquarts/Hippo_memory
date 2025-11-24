"""
===================================================================================
HIPPO_CA3 — CA3 Associative Memory with Global Inhibition & WTA
===================================================================================
지은이: GNJz | 발행: 2025.11.24

Pure CA3 Associative Memory with Advanced Control Mechanisms

📐 구현된 핵심 수식:

1️⃣ Hodgkin-Huxley Neuron Dynamics (Self-Implemented):
   C_m dV/dt = I_Na + I_K + I_L + I_ext
   
   where:
   I_Na = g_Na · m³ · h · (E_Na - V)
   I_K  = g_K · n⁴ · (E_K - V)
   I_L  = g_L · (E_L - V)
   
   Mode switching: rest ↔ active (RK4 integration in active mode)

2️⃣ Post-Tetanic Potentiation (PTP) Only:
   On spike:  PTP ← min(2.0, PTP + 0.05)
   Decay:     PTP ← max(1.0, PTP - 0.001)

3️⃣ Synaptic Transmission with Alpha Function:
   I_syn(t) = Σ Q · R · (t/τ) · exp(1 - t/τ)
   
   where:
   Q = Q_max · clip(R, 0.1, 2.0)    [PTP-modulated strength]
   τ = synaptic time constant

4️⃣ Theta Offset — Neuron-Specific Threshold Adjustment:
   V_activate_thresh(i) = V_rest + 1.5 + θ_offset(i)
   
   Pattern neurons:    θ_offset = 0.0 mV  (normal threshold)
   Background neurons: θ_offset = +5.0 mV (elevated threshold)
   
   → Background neurons are harder to activate

5️⃣ Self-Connections (Auto-Excitation):
   N_i → N_i (Q=20.0, self-loop)
   
   → Sustained activity maintenance
   → Attractor state stabilization

6️⃣ Cross-Connections (Pattern Network):
   Pattern neurons: N_i ↔ N_j (Q=20.0, strong)
   Background:      Random 8% (Q=3.0, weak)

7️⃣ Global Feedback Inhibition:
   I_inhib(t) = -N_active(t-1) · K_INHIB
   
   K_INHIB = 30.0 (balanced)

8️⃣ Dynamic Inhibition Scaling:
   Phase 1 (t < 1.0 ms):   No inhibition (allow initial activation)
   Phase 2 (1.0 < t < 4.0): 1.0× inhibition (pattern propagation)
   Phase 3 (t > 4.0 ms):    1.5× inhibition (strong suppression)

9️⃣ Winner-Take-All (WTA) — Two-Stage:
   Stage 1 (t > 2.5 ms): Top-3, normal suppression (-70 mV)
   Stage 2 (t > 4.0 ms): Top-3, aggressive suppression (-90 mV)

🔟 Pattern: N0, N5, N10 (3 neurons)
   Trigger: N0 only → Recall: N5, N10 (pattern completion)

===================================================================================
"""

"""
===================================================================================
📦 Note on Implementation
===================================================================================

This is a **standalone implementation** with its own HH neuron and synapse classes.
Unlike other files, it does NOT depend on the v3_event module.

Purpose:
  - Demonstrate CA3 auto-associative memory from first principles
  - Show advanced control mechanisms (theta offset, dynamic inhibition, WTA)
  - Serve as a complete reference for pattern completion

Unique Features:
  - Self-connections (N_i → N_i) for sustained activity
  - Theta offset for noise suppression
  - Dynamic inhibition scaling
  - Two-stage WTA (normal → aggressive)

===================================================================================
"""

# Qquarts co Present
# 지은이 : GNJz 
# 발행 2025.11.24

import numpy as np
import random

# =============================================================
# 0. CONFIG & SOLVER
# =============================================================

CONFIG = {
    "HH": {
        "V0": -70.0,          # 📐 Initial membrane potential [mV]
        "gNa": 120.0,         # 📐 Sodium conductance [mS/cm²]
        "gK": 36.0,           # 📐 Potassium conductance [mS/cm²]
        "gL": 0.3,            # 📐 Leak conductance [mS/cm²]
        "ENa": 50.0,          # 📐 Sodium reversal [mV]
        "EK": -77.0,          # 📐 Potassium reversal [mV]
        "EL": -54.4,          # 📐 Leak reversal [mV]
        "spike_thresh": 0.0,  # 📐 Spike threshold [mV]
    }
}


def rk4_step_quick(derivs, y, dt):
    """
    Runge-Kutta 4th order integrator
    
    📐 수식:
    k1 = f(y)
    k2 = f(y + 0.5·dt·k1)
    k3 = f(y + 0.5·dt·k2)
    k4 = f(y + dt·k3)
    y_new = y + (k1 + 2k2 + 2k3 + k4) / 6
    """
    k1 = dt * np.array(derivs(y))
    k2 = dt * np.array(derivs(y + 0.5 * k1))
    k3 = dt * np.array(derivs(y + 0.5 * k2))
    k4 = dt * np.array(derivs(y + k3))
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0


# =============================================================
# 1. HHSomaQuick (Self-Implemented HH Neuron)
# =============================================================

class HHSomaQuick:
    """
    Hodgkin-Huxley Neuron with Mode Switching
    
    📐 Modes:
    - rest:   Simplified dynamics (fast, for resting state)
    - active: Full HH dynamics with RK4 (accurate, for spiking)
    
    수식은 hippo_cortex.py와 유사하지만 mode switching 추가
    """
    
    def __init__(self, cfg):
        self.C_m = 1.0
        self.V = float(cfg["V0"])
        self.gNa, self.gK, self.gL = cfg["gNa"], cfg["gK"], cfg["gL"]
        self.ENa, self.EK, self.EL = cfg["ENa"], cfg["EK"], cfg["EL"]
        self.spike_thresh_v = cfg.get("spike_thresh", 0.0)
        self.m, self.h, self.n = 0.05, 0.60, 0.32
        self.mode = "rest"
        self.Vrest = self.EL
        self.V_activate_thresh = self.Vrest + 1.5  # 📐 Activation threshold
        self.active_remaining = 0.0
        self.REFRACTORY_TIME_MS = 5.0
        self.spike_flag = False
        self.I_syn_total = 0.0

    @staticmethod
    def _alpha_m(V):
        """📐 α_m(V) = 0.1·(V+40) / (1 - exp(-(V+40)/10))"""
        x = V + 40
        return 0.1*x/(1-np.exp(np.clip(-x/10,-50,50))) if abs(x)>1e-6 else 1.0
    
    @staticmethod
    def _beta_m(V):
        """📐 β_m(V) = 4.0·exp(-(V+65)/18)"""
        return 4.0*np.exp(np.clip(-(V+65)/18,-50,50))
    
    @staticmethod
    def _alpha_h(V):
        """📐 α_h(V) = 0.07·exp(-(V+65)/20)"""
        return 0.07*np.exp(np.clip(-(V+65)/20,-50,50))
    
    @staticmethod
    def _beta_h(V):
        """📐 β_h(V) = 1 / (1 + exp(-(V+35)/10))"""
        return 1.0/(1.0+np.exp(np.clip(-(V+35)/10,-50,50)))
    
    @staticmethod
    def _alpha_n(V):
        """📐 α_n(V) = 0.01·(V+55) / (1 - exp(-(V+55)/10))"""
        x = V + 55
        return 0.01*x/(1-np.exp(np.clip(-x/10,-50,50))) if abs(x)>1e-6 else 0.1
    
    @staticmethod
    def _beta_n(V):
        """📐 β_n(V) = 0.125·exp(-(V+65)/80)"""
        return 0.125*np.exp(np.clip(-(V+65)/80,-50,50))

    def _get_derivatives(self, y, I_ext):
        """📐 HH dynamics: dV/dt, dm/dt, dh/dt, dn/dt"""
        V, m, h, n = y
        INa = self.gNa*(m**3)*h*(self.ENa-V)
        IK = self.gK*(n**4)*(self.EK-V)
        IL = self.gL*(self.EL-V)
        dV = (INa + IK + IL + I_ext)/self.C_m
        am, bm = self._alpha_m(V), self._beta_m(V)
        ah, bh = self._alpha_h(V), self._beta_h(V)
        an, bn = self._alpha_n(V), self._beta_n(V)
        return [dV, am*(1-m)-bm*m, ah*(1-h)-bh*h, an*(1-n)-bn*n]

    def _rest_step(self, dt, I_ext):
        """📐 Simplified rest dynamics (Euler method)"""
        self.V += ((self.gL*(self.EL-self.V)+I_ext)/self.C_m)*dt
        V = self.V
        am, bm = self._alpha_m(V), self._beta_m(V)
        ah, bh = self._alpha_h(V), self._beta_h(V)
        an, bn = self._alpha_n(V), self._beta_n(V)
        # 📐 Exponential relaxation to steady state
        self.m += (am/(am+bm) - self.m)*(dt*(am+bm))
        self.h += (ah/(ah+bh) - self.h)*(dt*(ah+bh))
        self.n += (an/(an+bn) - self.n)*(dt*(an+bn))

    def add_synaptic_current(self, I):
        """📐 Accumulate synaptic input"""
        self.I_syn_total += I
        
    def get_total_synaptic_current(self):
        """📐 Get and reset synaptic current"""
        I = self.I_syn_total
        self.I_syn_total = 0.0
        return I

    def step(self, dt, I_ext=0.0):
        """
        Single timestep update with mode switching
        
        📐 Mode logic:
        rest → active:  if V > V_activate_thresh or I_ext > 5.0
        active → rest:  if V < V_activate_thresh and refractory expired
        """
        self.spike_flag = False
        I_tot = I_ext + self.I_syn_total
        self.I_syn_total = 0.0
        
        if self.mode == "active":
            # 📐 Full HH dynamics (RK4)
            y = np.array([self.V, self.m, self.h, self.n])
            y = rk4_step_quick(lambda x: self._get_derivatives(x, I_tot), y, dt)
            self.V, self.m, self.h, self.n = y
            self.active_remaining -= dt
            
            # 📐 Spike detection
            if self.V > self.spike_thresh_v and not self.spike_flag:
                self.spike_flag = True
                self.active_remaining = self.REFRACTORY_TIME_MS
                
            # 📐 Mode switching: active → rest
            if self.active_remaining <= 0 and self.V < self.V_activate_thresh:
                self.mode = "rest"
        else:
            # 📐 Simplified rest dynamics
            self._rest_step(dt, I_tot)
            
            # 📐 Mode switching: rest → active
            if self.V > self.V_activate_thresh or I_tot > 5.0:
                self.mode = "active"
                self.active_remaining = self.REFRACTORY_TIME_MS
        
        # 📐 Safety clipping
        self.V = np.clip(self.V, -90, 50)
        self.m, self.h, self.n = np.clip([self.m, self.h, self.n], 0, 1)
        
        return self.V

    def spiking(self):
        """📐 Spike flag (boolean)"""
        return self.spike_flag


# =============================================================
# 2. SynapseCore (Alpha Function)
# =============================================================

class SynapseCore:
    """
    Synapse with Alpha Function Conductance
    
    📐 수식:
    I_syn(t) = Q · (t/τ) · exp(1 - t/τ)    for t > delay
    
    where Q = Q_max · clip(R, 0.1, 2.0)  [PTP-modulated]
    """
    
    def __init__(self, post, delay=1.5, Q=15.0, tau=1.0):
        self.post = post
        self.delay = delay    # 📐 Synaptic delay [ms]
        self.Q_max = Q        # 📐 Maximum conductance
        self.tau = tau        # 📐 Time constant [ms]
        self.spikes = []      # 📐 Spike queue: [(time, Q)]
        
    def on_pre_spike(self, t, R):
        """
        Presynaptic spike event
        
        📐 수식: Q = Q_max · clip(R, 0.1, 2.0)
        
        Parameters:
            t: Spike time [ms]
            R: PTP value (modulates strength)
        """
        Q = self.Q_max * min(2.0, max(0.1, R))  # 📐 PTP modulation
        self.spikes.append((t, Q))
        # 📐 Garbage collection: remove old spikes
        self.spikes = [s for s in self.spikes if t - s[0] < 5*self.tau + self.delay]
        
    def deliver(self, t):
        """
        Deliver synaptic current at time t
        
        📐 수식:
        I = Σ Q · ((t-t_spike-delay)/τ) · exp(1 - (t-t_spike-delay)/τ)
        """
        I = 0.0
        for ts, Q in self.spikes:
            dt = t - (ts + self.delay)
            if dt > 0:
                # 📐 Alpha function
                I += Q * (dt/self.tau) * np.exp(1 - dt/self.tau)
        self.post.add_synaptic_current(I)


# =============================================================
# 3. LightNeuron (with Theta Offset)
# =============================================================

class LightNeuron:
    """
    Neuron with Adjustable Activation Threshold
    
    📐 Theta Offset Mechanism:
    V_activate_thresh = V_rest + 1.5 + θ_offset
    
    생물학적 의미:
    - Neuron-specific excitability control
    - Pattern neurons: θ = 0.0 (normal)
    - Noise neurons: θ = +5.0 (elevated, harder to activate)
    
    → Selective suppression of background activity
    """
    
    def __init__(self, name, theta_offset=0.0):
        self.name = name
        self.soma = HHSomaQuick(CONFIG["HH"])
        self.PTP = 1.0               # 📐 PTP state
        self.theta_offset = theta_offset  # 📐 Threshold adjustment [mV]
        
    def step(self, dt, I_ext=0.0):
        """
        Single timestep with theta offset application
        
        📐 수식: V_activate_thresh ← V_activate_thresh + θ_offset
        """
        # 📐 Apply theta offset
        original_thresh = self.soma.V_activate_thresh
        self.soma.V_activate_thresh = original_thresh + self.theta_offset
        
        # 📐 HH step
        self.soma.step(dt, I_ext)
        
        # 📐 Restore original threshold
        self.soma.V_activate_thresh = original_thresh
        
        # 📐 PTP update
        if self.soma.spiking():
            self.PTP = min(2.0, self.PTP + 0.05)
        else:
            self.PTP = max(1.0, self.PTP - 0.001)
            
        return self.soma.spiking(), self.PTP


# =============================================================
# 4. Winner-Take-All Mechanism
# =============================================================

def apply_wta(neurons, k=5, aggressive=False):
    """
    Winner-Take-All with Two Levels of Suppression
    
    📐 구현 수식:
    1) Sort by voltage: V_sorted = sort([V_1, ..., V_N], desc)
    2) Select winners: Winners = {i | V_i ∈ top-k}
    3) Suppress losers:
       Normal:     V_loser ← -70 mV
       Aggressive: V_loser ← -90 mV (stronger)
    
    Parameters:
        neurons: 뉴런 리스트
        k: 승자 수
        aggressive: True면 더 강력한 억제 (-90 mV)
    
    Returns:
        winners: 승자 인덱스 리스트
    """
    # 📐 수식: Sort by voltage
    voltages = [(i, n.soma.V) for i, n in enumerate(neurons)]
    voltages.sort(key=lambda x: x[1], reverse=True)
    
    # 📐 수식: Top-k selection
    winners = [idx for idx, _ in voltages[:k]]
    losers = [idx for idx, _ in voltages[k:]]
    
    # 📐 수식: Suppression level
    suppress_V = -90.0 if aggressive else -70.0
    
    for idx in losers:
        if neurons[idx].soma.V > -60.0:  # Only if active
            neurons[idx].soma.V = suppress_V
            neurons[idx].soma.mode = "rest"
            neurons[idx].soma.spike_flag = False
            neurons[idx].soma.I_syn_total = 0.0
    
    return winners


# =============================================================
# 5. CA3 Associative Network with Advanced Controls
# =============================================================

def run_ca3_inhib():
    """
    CA3 Auto-Associative Memory Simulation
    
    📐 핵심 메커니즘:
    
    1) Self-Connections (Auto-Excitation):
       N_i → N_i (Q=20.0)
       
       → Sustained activity (attractor stabilization)
    
    2) Cross-Connections (Pattern Network):
       Pattern: N_i ↔ N_j (Q=20.0, strong)
       Background: Random 8% (Q=3.0, weak)
    
    3) Theta Offset:
       Pattern neurons: θ = 0.0 mV
       Others: θ = +5.0 mV
       
       → Selective suppression
    
    4) Dynamic Inhibition:
       t < 1.0 ms:  0.0× (no inhibition)
       1.0-4.0 ms:  1.0× (normal)
       t > 4.0 ms:  1.5× (strong)
    
    5) Two-Stage WTA:
       t > 2.5 ms: Top-3, normal (-70 mV)
       t > 4.0 ms: Top-3, aggressive (-90 mV)
    
    6) Pattern: N0, N5, N10 (3 neurons)
       Trigger: N0 → Recall: N5, N10
    """
    
    # 🎲 Reproducibility
    random.seed(42)
    np.random.seed(42)
    
    N = 20
    dt = 0.1
    
    # 📐 Theta offset: +5.0 mV for background neurons
    pattern_indices = [0, 5, 10]
    theta_offsets = np.zeros(N)
    for i in range(N):
        if i not in pattern_indices:
            theta_offsets[i] = 5.0  # 📐 Elevated threshold
    
    neurons = [LightNeuron(f"N{i}", theta_offset=theta_offsets[i]) for i in range(N)]
    
    print("\n" + "="*70)
    print("🧠 CA3 ASSOCIATIVE MEMORY SIMULATION")
    print("="*70)
    print("Network Size: 20 neurons")
    print(f"Pattern: N0 ↔ N5 ↔ N10 (Self-sustaining attractor)")
    print("-"*70)
    print("🔄 Self-Connections:")
    print(f"   • N0 → N0, N5 → N5, N10 → N10 (Q=20.0)")
    print(f"   • Purpose: Maintain sustained activity")
    print("-"*70)
    print("🔌 Control Mechanism (High Performance):")
    print(f"   • K_INHIB = 30.0 (balanced inhibition)")
    print(f"   • Theta offset = +5.0mV (noise neurons)")
    print(f"   • Dynamic brake: 0x→1.0x→1.5x")
    print(f"   • WTA: 2-stage (2.5ms normal, 4.0ms aggressive)")
    print("-"*70)
    print("🎯 Strategy: Maximum Accuracy")
    print(f"   • Pattern neurons: Theta=0mV, Q=20.0")
    print(f"   • Noise neurons: Theta=+5.0mV, Q=3.0")
    print(f"   • Target: 75%+ accuracy (0-1 noise neuron)")
    print("="*70)
    
    # --------------------------------------------------------
    # 연결성: Random Recurrent (8%) + Pattern + Self
    # --------------------------------------------------------
    synapses = []
    adj_matrix = np.zeros((N, N))
    
    # 📐 Random background connections (8%, Q=3.0)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if random.random() < 0.08:
                syn = SynapseCore(neurons[j].soma, Q=3.0)  # Weak background
                synapses.append((i, syn))
                adj_matrix[i][j] = 1

    # 📐 Self-connections (N_i → N_i, Q=20.0)
    for i in pattern_indices:
        syn = SynapseCore(neurons[i].soma, Q=20.0)  # Strong self-excitation
        synapses.append((i, syn))
        adj_matrix[i][i] = 1
        print(f"🔄 Self-Link: N{i} -> N{i} (Sustained activity)")
    
    # 📐 Cross-connections (Pattern: N_i ↔ N_j, Q=20.0)
    for i in pattern_indices:
        for j in pattern_indices:
            if i == j: continue  # Self already added
            if adj_matrix[i][j] == 0:
                syn = SynapseCore(neurons[j].soma, Q=20.0)  # Strong pattern links
                synapses.append((i, syn))
                adj_matrix[i][j] = 1
                print(f"Pattern Link: N{i} -> N{j}")

    print("\n" + "="*70)
    print("🔗 NETWORK CONNECTIVITY")
    print("="*70)
    print(f"Total Synapses: {len(synapses)}")
    print(f"Connection Density: {len(synapses)/(N*(N-1))*100:.1f}%")
    print(f"Pattern Connections:")
    print(f"  • Self-connections: 3 (N0↔N0, N5↔N5, N10↔N10) - Q=20.0")
    print(f"  • Cross-connections: 6 (N0↔N5, N5↔N10, N10↔N0) - Q=20.0")
    print(f"Background Connections: ~{len(synapses)-9} (Q=3.0, weak)")
    print("="*70)

    # 📐 Global Inhibition Parameter
    K_INHIB = 30.0  # Balanced inhibition
    
    # --------------------------------------------------------
    # PHASE 1: LEARNING
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("📚 LEARNING PHASE: Co-activation of N0, N5, N10")
    print("="*70)
    print("Duration: 50ms | Stimulus: 200.0 pA (0-10ms)")
    T_learn = 50.0
    last_active_count = 0
    
    for t in np.arange(0, T_learn, dt):
        current_active = 0
        
        # 📐 1. Synapse delivery
        for _, syn in synapses:
            syn.deliver(t)
        
        # 📐 2. Global inhibition (feedback)
        I_inhib = -1.0 * last_active_count * K_INHIB
        
        # 📐 3. Neuron update
        for i in range(N):
            I_stim = 200.0 if (i in pattern_indices and t < 10.0) else 0.0
            sp, ptp = neurons[i].step(dt, I_ext=I_stim + I_inhib)
            
            if sp:
                current_active += 1
                # 📐 Synaptic transmission
                for pre_id, syn in synapses:
                    if pre_id == i:
                        syn.on_pre_spike(t, ptp)
        
        last_active_count = current_active

    print("\n✅ Learning Complete!")
    print("-" * 70)
    print("🧬 Synaptic Plasticity (PTP) Status:")
    for i in [0, 5, 10]:
        ptp = neurons[i].PTP
        bar = "█" * int(ptp * 20)
        print(f"   N{i:2d}: {bar:40s} {ptp:.3f}")
    print("="*70)

    # --------------------------------------------------------
    # PHASE 2: RESET
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("🔄 RESET PHASE: Restoring resting state")
    print("="*70)
    
    for t in np.arange(0, 50, dt):
        for n in neurons:
            n.step(dt, 0)
        for _, s in synapses:
            s.deliver(t)
    
    for n in neurons:
        n.soma.V = -70.0
        n.soma.m, n.soma.h, n.soma.n = 0.05, 0.60, 0.32
        n.soma.spike_flag = False
        n.soma.I_syn_total = 0.0
        n.soma.mode = "rest"
        n.soma.active_remaining = 0.0
        
    for _, s in synapses:
        s.spikes = []
    
    last_active_count = 0

    print("✅ Reset Complete!\n")
    
    # --------------------------------------------------------
    # PHASE 3: RECALL
    # --------------------------------------------------------
    print("="*70)
    print("🧪 RECALL PHASE: Pattern Completion Test")
    print("="*70)
    print("Trigger: N0 only (200.0 pA, 0-0.8ms)")
    print("Expected: N0 → N5, N10 activation (pattern completion)")
    print("-"*70)
    
    T_recall = 20.0  # Short observation window
    logs = []

    for t in np.arange(0, T_recall, dt):
        current_active = 0
        active_ids = []
        
        # 📐 1. Synapse delivery
        for _, syn in synapses:
            syn.deliver(t)
        
        # 📐 2. Dynamic inhibition scaling
        if t > 4.0:  # Strong suppression after pattern completion
            I_inhib = -1.5 * last_active_count * K_INHIB
        elif t > 1.0:  # Normal inhibition during propagation
            I_inhib = -1.0 * last_active_count * K_INHIB
        else:  # No inhibition (allow initial activation)
            I_inhib = 0.0
        
        # 📐 3. Neuron update
        for i in range(N):
            I_stim = 200.0 if (i == 0 and t < 0.8) else 0.0
            sp, ptp = neurons[i].step(dt, I_ext=I_stim + I_inhib)
            
            if sp:
                current_active += 1
                active_ids.append(i)
                # 📐 Synaptic transmission
                for pre_id, syn in synapses:
                    if pre_id == i:
                        syn.on_pre_spike(t, ptp)
        
        # 📐 4. Two-stage WTA
        if t > 4.0:  # Aggressive suppression
            winners = apply_wta(neurons, k=3, aggressive=True)
        elif t > 2.5:  # Normal suppression
            winners = apply_wta(neurons, k=3, aggressive=False)
        
        last_active_count = current_active
        if active_ids:
            logs.append((t, active_ids))

    # --------------------------------------------------------
    # 결과 분석 (시각화)
    # --------------------------------------------------------
    print("\n" + "="*70)
    print("🔥 NEURAL ACTIVITY HEATMAP (First 10ms)")
    print("="*70)
    print("Time |  N0  N1  N2  N3  N4  N5  N6  N7  N8  N9 N10 N11 N12 N13 N14 N15 N16 N17 N18 N19")
    print("-" * 70)
    
    # 📐 Activity matrix
    activity_matrix = {}
    for t, ids in logs:
        if t <= 10.0:
            time_bin = round(t, 1)
            if time_bin not in activity_matrix:
                activity_matrix[time_bin] = [False] * N
            for idx in ids:
                activity_matrix[time_bin][idx] = True
    
    # 📐 Heatmap visualization
    pattern_neurons_set = set([0, 5, 10])
    for t in sorted(activity_matrix.keys())[::5]:  # 0.5ms intervals
        row = f"{t:4.1f} | "
        for i in range(N):
            if activity_matrix[t][i]:
                if i in pattern_neurons_set:
                    row += " 🔴"  # Pattern neuron
                else:
                    row += " 🟡"  # Noise neuron
            else:
                row += "  ·"
        print(row)
    
    print("="*70)
    print("Legend: 🔴 Pattern neuron  🟡 Noise neuron  · Silent")
    print("="*70)

    # 📐 Final result check
    recalled = set()
    for _, ids in logs:
        for i in ids:
            recalled.add(i)
    
    print("\n" + "="*70)
    print("📊 PATTERN COMPLETION ANALYSIS")
    print("="*70)
    
    print("\n🧠 Neuron Status:")
    print("-" * 70)
    for i in range(N):
        if i in [0, 5, 10]:
            status = "🔴 PATTERN" if i in recalled else "❌ MISSING"
            ptp = neurons[i].PTP
            print(f"  N{i:2d}: {status:12s} | PTP={ptp:.3f}")
        elif i in recalled:
            print(f"  N{i:2d}: 🟡 NOISE     | (unwanted activation)")
    
    print("-" * 70)
    print(f"\n🎯 Target Pattern: N0, N5, N10")
    print(f"✓ Recalled Neurons: {sorted(list(recalled))}")
    
    # 📐 Evaluation
    pattern_ok = (0 in recalled) and (5 in recalled) and (10 in recalled)
    noise_neurons = [n for n in recalled if n not in [0, 5, 10]]
    noise_count = len(noise_neurons)
    
    print("\n" + "="*70)
    if pattern_ok:
        if noise_count == 0:
            print("🏆 PERFECT: Complete pattern recall with ZERO noise!")
            print("="*70)
        elif noise_count < 3:
            print(f"✅ EXCELLENT: Pattern completed with minimal noise (+{noise_count})")
            print(f"   Noise neurons: {noise_neurons}")
            print("="*70)
        elif noise_count < 5:
            print(f"✅ SUCCESS: Pattern completed with acceptable noise (+{noise_count})")
            print(f"   Noise neurons: {noise_neurons}")
            print("="*70)
        else:
            print(f"⚠️ PARTIAL: Pattern completed but noisy (+{noise_count} neurons)")
            print(f"   Noise neurons: {noise_neurons}")
            print("   💡 Tip: Increase K_INHIB or theta_offset to suppress noise.")
            print("="*70)
    else:
        print("❌ FAILED: Pattern not fully recalled")
        print("-" * 70)
        missing = []
        if 0 not in recalled:
            missing.append("N0")
        if 5 not in recalled:
            missing.append("N5")
        if 10 not in recalled:
            missing.append("N10")
        print(f"   Missing neurons: {', '.join(missing)}")
        print("   💡 Tip: Decrease K_INHIB or theta_offset")
        print("="*70)
    
    # 📐 Statistics
    total_spikes = len(logs)
    duration = logs[-1][0] - logs[0][0] if logs else 0
    print(f"\n📈 Statistics:")
    print(f"   Total spike events: {total_spikes}")
    print(f"   Activity duration: {duration:.1f}ms")
    print(f"   Recall accuracy: {3/(3+noise_count)*100:.1f}% ({3}/{3+noise_count})")
    print("="*70)


if __name__ == "__main__":
    run_ca3_inhib()
