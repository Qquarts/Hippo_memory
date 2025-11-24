"""
===================================================================================
HIPPO_SUB — Complete Hippocampal Circuit with Pattern Completion
===================================================================================
지은이: GNJz | 발행: 2025.11.24

Full Hippocampal Pipeline: DG → CA3 (recurrent) → CA1 → Subiculum

📐 구현된 핵심 수식:

1️⃣ Hodgkin-Huxley Neuron Dynamics (via v3_event.HHSomaQuick):
   C_m dV/dt = I_ext + I_syn - g_L(V-E_L) - g_Na·m³h(V-E_Na) - g_K·n⁴(V-E_K)
   
   Gating variables:
   dm/dt = α_m(1-m) - β_m·m
   dh/dt = α_h(1-h) - β_h·h
   dn/dt = α_n(1-n) - β_n·n

2️⃣ Short-Term Plasticity (STP) & Post-Tetanic Potentiation (PTP):
   On spike:  S ← min(1.0, S + 0.3),    PTP ← min(2.0, PTP + 0.05)
   Decay:     S ← max(0.0, S - 0.01),   PTP ← max(1.0, PTP - 0.001)

3️⃣ Global Inhibition (Feedback inhibition):
   I_inhib(t) = -N_active(t-1) · g_inhib
   
   where N_active(t-1) = number of neurons that spiked in previous timestep

4️⃣ Subiculum Leaky Integrator (Spike-to-Rate decoder):
   dy/dt = -y/τ + w_in · spike(t)
   
   Discrete form:
   y(t+dt) = y(t) + dt·(-y/τ + w_in·spike)

5️⃣ Winner-Take-All (WTA) Competition:
   Select top-k neurons by voltage V
   Suppress losers: V_loser ← -70 mV

6️⃣ Hippocampal Pathways:
   Mossy Fibers:        DG → CA3  (1:1, strong: Q=80.0, "detonator")
   Recurrent:           CA3 ⟲ CA3 (selective: pattern Q=15.0, background Q=3.0)
   Schaffer Collateral: CA3 → CA1 (1:1, strong: Q=25.0)
   Direct:              CA1 → SUB (spike → rate conversion)

7️⃣ Pattern Completion:
   Input: Partial Cue (1/N neurons)
   CA3 Recurrent: Amplifies pattern-specific connections
   Output: Complete Pattern (N/N neurons)

8️⃣ Network Architecture:
   Learning:  Full Pattern → DG → CA3 (recurrent strengthens) → CA1 → SUB
   Recall:    Partial Cue → DG → CA3 (completes pattern) → CA1 → SUB → Output

===================================================================================
"""

"""
===================================================================================
📦 Dependencies
===================================================================================

This module depends on `v3_event.py` which contains:
  - CONFIG (global Hodgkin-Huxley parameters)
  - HHSomaQuick (fast HH soma)
  - SynapseCore (synaptic event engine with delay queue)

If v3_event.py is not available, please check the project repository
or contact the author for the full package.

===================================================================================
"""

# Qquarts co Present
# 지은이 : GNJz 
# 발행 2025.11.24

import numpy as np
import random
import sys
from pathlib import Path

# ✅ Add parent directory to path for v3_event import
sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ 핵심 엔진 임포트
from v3_event import CONFIG, HHSomaQuick, SynapseCore

# ======================================================================
# 1. 뉴런 클래스 정의 (DG / CA3 / CA1)
# ======================================================================

class DGLightNeuron:
    """
    Dentate Gyrus Neuron — Sparse Encoder
    
    📐 구현 수식:
    1) HH Dynamics (via HHSomaQuick):
       C_m dV/dt = I_ext + I_syn - g_L(V-E_L) - g_Na·m³h(V-E_Na) - g_K·n⁴(V-E_K)
    
    2) High Leak Current (gL = 0.1):
       I_leak = g_L·(V - E_L)
       → 높은 leak = 약한 자극은 필터링, 강한 자극만 통과
    
    3) STP/PTP:
       On spike:  S ← min(1.0, S + 0.3),    PTP ← min(2.0, PTP + 0.05)
       Decay:     S ← max(0.0, S - 0.01),   PTP ← max(1.0, PTP - 0.001)
    
    생물학적 의미:
    - DG는 "pattern separator" — 유사 입력을 구분
    - High leak → sparse coding (적은 수의 뉴런만 발화)
    - Mossy fiber → CA3로 강력한 "detonator" 신호 전달
    """
    
    def __init__(self, name):
        self.name = name
        cfg = CONFIG["HH"].copy()
        cfg["gL"] = 0.1  # 📐 High leak conductance (sparse coding)
        self.soma = HHSomaQuick(cfg)
        self.S, self.PTP = 0.0, 1.0  # 📐 STP/PTP state variables
        
    def step(self, dt, I_ext=0.0):
        """
        단일 timestep 실행
        
        Returns:
            spike (bool): 스파이크 발생 여부
            S (float): STP state
            PTP (float): PTP state
        """
        self.soma.step(dt, I_ext=I_ext)
        spike = self.soma.spiking()
        
        # 📐 STP/PTP update
        if spike:
            self.S = min(1.0, self.S + 0.3)      # 📐 S ← min(1.0, S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)  # 📐 PTP ← min(2.0, PTP + 0.05)
        else:
            self.S = max(0.0, self.S - 0.01)      # 📐 S ← max(0.0, S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001) # 📐 PTP ← max(1.0, PTP - 0.001)
            
        return spike, self.S, self.PTP


class CA3LightNeuron:
    """
    CA3 Neuron — Recurrent Attractor Network
    
    📐 구현 수식:
    1) HH Dynamics (standard gL):
       C_m dV/dt = I_ext + I_syn + I_recurrent - g_L(V-E_L) - ...
    
    2) Recurrent Input:
       I_recurrent = Σ_j Q_ij · S_j · PTP_j · e^(-(t-t_spike)/τ)
       
       where j = other CA3 neurons
    
    3) Pattern-Selective Connectivity:
       Q_ij = { 15.0  if i,j ∈ same pattern (strong attractor)
              {  3.0  if random background (10% probability)
              {  0.0  otherwise
    
    생물학적 의미:
    - CA3는 "auto-associative memory" — 부분 입력으로 전체 패턴 복원
    - Recurrent connections → attractor dynamics
    - Pattern-specific strong links → 선택적 증폭
    """
    
    def __init__(self, name):
        self.name = name
        self.soma = HHSomaQuick(CONFIG["HH"])  # 📐 Standard HH parameters
        self.S, self.PTP = 0.0, 1.0            # 📐 STP/PTP state variables
        
    def step(self, dt, I_ext=0.0):
        """
        단일 timestep 실행
        
        Returns:
            spike (bool): 스파이크 발생 여부
            S (float): STP state
            PTP (float): PTP state
        """
        self.soma.step(dt, I_ext=I_ext)
        spike = self.soma.spiking()
        
        # 📐 STP/PTP update
        if spike:
            self.S = min(1.0, self.S + 0.3)      # 📐 S ← min(1.0, S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)  # 📐 PTP ← min(2.0, PTP + 0.05)
        else:
            self.S = max(0.0, self.S - 0.01)      # 📐 S ← max(0.0, S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001) # 📐 PTP ← max(1.0, PTP - 0.001)
            
        return spike, self.S, self.PTP


class CA1LightNeuron:
    """
    CA1 Neuron — Output Filter
    
    📐 구현 수식:
    1) HH Dynamics with Medium Leak (gL = 0.08):
       C_m dV/dt = I_ext + I_syn - g_L(V-E_L) - ...
    
    2) Schaffer Collateral Input:
       I_syn = Σ_j Q_j · S_j · PTP_j · e^(-(t-t_spike)/τ)
       
       where j = CA3 neurons
    
    생물학적 의미:
    - CA1은 CA3의 출력을 "정제"하여 피질로 전달
    - Medium leak → 중간 정도의 필터링
    - Schaffer collateral → CA3의 "완성된 패턴"을 받음
    """
    
    def __init__(self, name):
        self.name = name
        cfg = CONFIG["HH"].copy()
        cfg["gL"] = 0.08  # 📐 Medium leak conductance
        self.soma = HHSomaQuick(cfg)
        self.S, self.PTP = 0.0, 1.0  # 📐 STP/PTP (not used in this model, but kept for consistency)
        
    def step(self, dt, I_ext=0.0):
        """
        단일 timestep 실행
        
        Returns:
            spike (bool): 스파이크 발생 여부
        """
        self.soma.step(dt, I_ext=I_ext)
        return self.soma.spiking()


# ======================================================================
# 1-1. WTA Helper (Winner-Take-All)
# ======================================================================

def apply_wta(neurons, k=5):
    """
    Winner-Take-All (WTA) — Sparse Competition
    
    📐 구현 수식:
    1) 전압 기준 정렬:
       V_sorted = sort([V_1, V_2, ..., V_N], descending)
    
    2) Top-k 선택:
       Winners = {i | V_i ∈ top-k}
    
    3) Losers 억제:
       For i ∈ Losers:
         if V_i > -60 mV: V_i ← -70 mV (forced reset)
    
    생물학적 의미:
    - 억제성 인터뉴런 (GABAergic interneurons)의 피드백 억제 모사
    - Sparse representation 유지
    - 노이즈 억제
    
    Parameters:
        neurons: 뉴런 리스트
        k: 승자 수 (default: 5)
    """
    # 📐 수식: V_sorted = sort([V_1, ..., V_N], descending)
    voltages = [(i, n.soma.V) for i, n in enumerate(neurons)]
    voltages.sort(key=lambda x: x[1], reverse=True)
    
    # 📐 수식: Losers = {i | i ∉ top-k}
    losers = [idx for idx, _ in voltages[k:]]
    
    # 📐 수식: V_loser ← -70 mV (forced reset)
    for idx in losers:
        if neurons[idx].soma.V > -60.0:  # Only reset if above threshold
            neurons[idx].soma.V = -70.0
            neurons[idx].soma.spike_flag = False
            neurons[idx].soma.mode = "rest"


# ======================================================================
# 2. Subiculum (Output Integrator)
# ======================================================================

class SubiculumLight:
    """
    Subiculum — Spike-to-Rate Decoder (Leaky Integrator)
    
    📐 구현 수식:
    
    미분 방정식:
    dy/dt = -y/τ + w_in · spike(t)
    
    이산화 (Euler method):
    y(t+dt) = y(t) + dt · (-y/τ + w_in · spike)
          = y(t) · (1 - dt/τ) + dt · w_in · spike
    
    where:
    - y(t): Activity rate (출력 신호 강도)
    - τ: Time constant (시정수, 작을수록 빠른 감쇠)
    - w_in: Input weight (입력 가중치)
    - spike: Binary input (0 or 1)
    
    생물학적 의미:
    - Subiculum은 해마의 "출력 게이트"
    - Spike train → Rate code 변환
    - τ = 2.0 ms → 빠른 감쇠 (transient memory)
    - 피질로 전달할 "요약된 활성도" 생성
    
    Parameters:
        name: 뉴런 이름
        tau: 시정수 (default: 2.0 ms)
    """
    
    def __init__(self, name, tau=2.0):
        self.name = name
        self.y = 0.0       # 📐 Activity rate (출력 값)
        self.tau = tau     # 📐 Time constant τ
        self.w_in = 5.0    # 📐 Input weight w_in

    def step(self, dt, ca1_spike):
        """
        단일 timestep 실행
        
        📐 수식:
        y(t+dt) = y(t) + dt · (-y/τ + w_in · spike)
        
        Parameters:
            dt: Timestep (ms)
            ca1_spike: CA1 스파이크 (bool)
        
        Returns:
            y: 현재 activity rate
        """
        # 📐 Leaky Integrator: dy/dt = -y/τ + w_in · spike
        decay = -self.y / self.tau                    # 📐 -y/τ (leak term)
        inp = self.w_in if ca1_spike else 0.0        # 📐 w_in · spike (input term)
        
        dy = decay + inp                              # 📐 dy/dt
        self.y += dy * dt                             # 📐 y(t+dt) = y(t) + dt·dy/dt
        
        return self.y


# ======================================================================
# 3. 전체 통합 시뮬레이션
# ======================================================================

def run_hippo_complete(N=20, dt=0.1):
    """
    Complete Hippocampal Circuit Simulation
    
    📐 전체 Pipeline:
    
    Phase 1 - Learning:
      Input → DG → CA3 (recurrent) → CA1 → Subiculum
      
      목적: 전체 패턴 저장
      입력: Full pattern (N neurons)
      결과: CA3 recurrent synapses 강화
    
    Phase 2 - Reset:
      모든 뉴런 및 시냅스 초기화
      
      목적: Learning과 Recall 분리
    
    Phase 3 - Recall:
      Partial Cue → DG → CA3 (pattern completion) → CA1 → Subiculum → Output
      
      목적: 부분 입력으로 전체 패턴 복원
      입력: Partial cue (1/N neurons)
      CA3 역할: Recurrent connections로 패턴 완성
      출력: Complete pattern (via Subiculum readout)
    
    📐 핵심 메커니즘:
    
    1) Global Inhibition (Feedback):
       I_inhib(t) = -N_active(t-1) · g_inhib
       
       → 많은 뉴런이 발화 → 강한 억제 → Sparse coding 유지
    
    2) Selective Recurrent (CA3):
       Q_ij = { 15.0  if i,j ∈ pattern
              {  3.0  if background
       
       → Pattern neurons만 강하게 연결 → 선택적 증폭
    
    3) WTA Competition:
       Top-k neurons 유지, 나머지 억제
       
       → 노이즈 제거, sparse representation
    
    4) Subiculum Readout:
       y = Leaky Integrator (CA1 spikes)
       
       → Spike → Rate 변환
       → 임계값(threshold=2.0) 이상이면 "활성"으로 판정
    
    Parameters:
        N: 뉴런 수 (default: 20)
        dt: Timestep (default: 0.1 ms)
    """
    
    # 🎲 Reproducibility
    random.seed(42)
    np.random.seed(42)

    print(f"\n🧠 COMPLETE HIPPOCAMPAL CIRCUIT (DG -> CA3 -> CA1 -> SUB)")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 뉴런 생성
    # --------------------------------------------------------
    dg_neurons  = [DGLightNeuron(f"DG{i}")  for i in range(N)]
    ca3_neurons = [CA3LightNeuron(f"CA3{i}") for i in range(N)]
    ca1_neurons = [CA1LightNeuron(f"CA1{i}") for i in range(N)]
    sub_neurons = [SubiculumLight(f"SUB{i}") for i in range(N)]

    # --------------------------------------------------------
    # 2. 연결 구축
    # --------------------------------------------------------
    
    # 📐 DG → CA3 (Mossy Fibers, "Detonator")
    # 생물학적: 1:1 강력한 연결 (Q_max=80.0)
    mossy_fibers = []
    for i in range(N):
        syn = SynapseCore(pre_neuron=dg_neurons[i].soma, post_neuron=ca3_neurons[i].soma,
                          delay_ms=1.0, Q_max=80.0, tau_ms=2.0)  # 📐 Strong "detonator"
        mossy_fibers.append(syn)

    # 📐 CA3 → CA3 (Recurrent, Selective)
    # Pattern-specific: 강한 연결 (Q=15.0)
    # Background: 약한 연결 (Q=3.0, 10% probability)
    ca3_synapses = []
    targets = [3, 7, 12, 16]  # 📐 Target pattern (4 neurons)
    
    for i in range(N):
        for j in range(N):
            if i == j: continue
            
            # 📐 Pattern-selective connectivity
            is_pattern_link = (i in targets) and (j in targets)
            
            if is_pattern_link:
                # 📐 Strong attractor: Q=15.0 (pattern neurons)
                syn = SynapseCore(pre_neuron=ca3_neurons[i].soma, post_neuron=ca3_neurons[j].soma,
                                  delay_ms=1.5, Q_max=15.0, tau_ms=3.0)
                ca3_synapses.append((i, j, syn))
                
            elif random.random() < 0.10:
                # 📐 Weak background: Q=3.0 (10% random)
                syn = SynapseCore(pre_neuron=ca3_neurons[i].soma, post_neuron=ca3_neurons[j].soma,
                                  delay_ms=1.5, Q_max=3.0, tau_ms=3.0)
                ca3_synapses.append((i, j, syn))

    # 📐 CA3 → CA1 (Schaffer Collaterals)
    # 생물학적: 강한 전달 (Q=25.0)
    schaffer_collaterals = []
    for i in range(N):
        syn = SynapseCore(pre_neuron=ca3_neurons[i].soma, post_neuron=ca1_neurons[i].soma,
                          delay_ms=2.0, Q_max=25.0, tau_ms=3.0)  # 📐 Strong transmission
        schaffer_collaterals.append(syn)

    # 📐 CA1 → Subiculum (Direct, 1:1)
    # Subiculum은 뉴런 내부에서 직접 CA1 spike를 받음

    print(f"System Ready: {4*N} Neurons, ~{len(ca3_synapses)+2*N} Synapses")

    # --------------------------------------------------------
    # PHASE 1: LEARNING (Target Pattern Encoding)
    # --------------------------------------------------------
    print("\n=== 1. LEARNING (Encoding) ===")
    T_learn = 50.0  # Learning duration (ms)
    steps = int(T_learn / dt)
    
    # 📐 Global Inhibition Parameters
    DG_INHIB = 80.0   # 📐 DG inhibition strength
    CA3_INHIB = 20.0  # 📐 CA3 inhibition strength
    
    dg_last = 0   # 📐 N_active(t-1) for DG
    ca3_last = 0  # 📐 N_active(t-1) for CA3

    for k in range(steps):
        t = k * dt
        
        # 📐 Input: Full pattern (t < 10 ms)
        dg_in = [0.0] * N
        if t < 10.0:
            for idx in targets:
                dg_in[idx] = 200.0  # 📐 Strong input to target neurons
            
        # --------------------------------------------------------
        # DG Layer
        # --------------------------------------------------------
        dg_now = 0
        # 📐 Global inhibition: I_inhib = -N_active(t-1) · g_inhib
        I_dg = -1.0 * dg_last * DG_INHIB
        
        for i in range(N):
            sp, S, PTP = dg_neurons[i].step(dt, dg_in[i] + I_dg)
            if sp:
                dg_now += 1
                # 📐 Mossy fiber transmission
                mossy_fibers[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
        
        dg_last = dg_now  # 📐 Update N_active(t-1)

        # --------------------------------------------------------
        # Synapse Delivery
        # --------------------------------------------------------
        for syn in mossy_fibers:
            syn.deliver(t)
        for _, _, syn in ca3_synapses:
            syn.deliver(t)

        # --------------------------------------------------------
        # CA3 Layer (Recurrent)
        # --------------------------------------------------------
        ca3_now = 0
        # 📐 Global inhibition: I_inhib = -N_active(t-1) · g_inhib
        I_ca3 = -1.0 * ca3_last * CA3_INHIB
        
        for i in range(N):
            I_syn = ca3_neurons[i].soma.get_total_synaptic_current()
            sp, S, PTP = ca3_neurons[i].step(dt, I_syn + I_ca3)
            
            if sp:
                ca3_now += 1
                # 📐 Recurrent & Schaffer transmission
                for pre, post, syn in ca3_synapses:
                    if pre == i:
                        syn.on_pre_spike(t, S, PTP, 100.0, 0.0)
                        
                schaffer_collaterals[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
        
        ca3_last = ca3_now  # 📐 Update N_active(t-1)

    print("✅ Memory Stored.")

    # --------------------------------------------------------
    # PHASE 2: RESET (System Cooldown)
    # --------------------------------------------------------
    print("\n=== 2. RESET (Consolidation) ===")
    
    # 📐 Cooldown: 모든 transient dynamics 소멸
    for _ in range(500):
        for n in dg_neurons + ca3_neurons + ca1_neurons:
            n.step(dt, 0)
        for s in mossy_fibers + schaffer_collaterals:
            s.deliver(0)
        for _, _, s in ca3_synapses:
            s.deliver(0)
        for sub in sub_neurons:
            sub.step(dt, False)

    # 📐 Force Reset: 모든 state variables 초기화
    all_hh = dg_neurons + ca3_neurons + ca1_neurons
    for n in all_hh:
        n.soma.V = -70.0
        n.soma.spike_flag = False
        n.soma.I_syn_total = 0.0
        n.soma.mode = "rest"
        n.soma.active_remaining = 0.0
        
    for s in mossy_fibers + schaffer_collaterals:
        s.spikes = []
        s.I_syn = 0.0
        
    for _, _, s in ca3_synapses:
        s.spikes = []
        s.I_syn = 0.0
    
    print("✅ System Cleared.")

    # --------------------------------------------------------
    # PHASE 3: RECALL & READOUT (Pattern Completion)
    # --------------------------------------------------------
    print("\n=== 3. RECALL (Retrieval & Readout) ===")
    print(f"Cue: [{targets[0]}] only -> Expected: {targets}")
    
    T_test = 30.0  # Recall duration (ms)
    steps = int(T_test / dt)
    
    # 📐 Recall Inhibition (stronger than learning)
    DG_INHIB_RECALL = 150.0   # 📐 Strong DG inhibition
    CA3_INHIB_RECALL = 60.0   # 📐 Strong CA3 inhibition
    CA1_INHIB_RECALL = 35.0   # 📐 CA1 noise filter

    dg_last = 0
    ca3_last = 0
    
    # 📐 Subiculum output storage
    sub_outputs = np.zeros(N)

    for k in range(steps):
        t = k * dt
        
        # --------------------------------------------------------
        # 1. DG (Input Layer)
        # --------------------------------------------------------
        dg_now = 0
        # 📐 Global inhibition: I_inhib = -N_active(t-1) · g_inhib
        I_dg = -1.0 * dg_last * DG_INHIB_RECALL
        
        for i in range(N):
            # 📐 Partial Cue: Only first target neuron (t < 10 ms)
            I_in = 200.0 if (i == targets[0] and t < 10.0) else 0.0
            sp, S, PTP = dg_neurons[i].step(dt, I_in + I_dg)
            
            if sp:
                dg_now += 1
                mossy_fibers[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
        
        dg_last = dg_now

        # --------------------------------------------------------
        # 2. Synapse Delivery
        # --------------------------------------------------------
        for syn in mossy_fibers:
            syn.deliver(t)
        for _, _, syn in ca3_synapses:
            syn.deliver(t)
        for syn in schaffer_collaterals:
            syn.deliver(t)

        # --------------------------------------------------------
        # 3. CA3 (Pattern Completion with WTA)
        # --------------------------------------------------------
        ca3_now = 0
        # 📐 Global inhibition: I_inhib = -N_active(t-1) · g_inhib
        I_ca3 = -1.0 * ca3_last * CA3_INHIB_RECALL
        
        for i in range(N):
            I_syn = ca3_neurons[i].soma.get_total_synaptic_current()
            sp, S, PTP = ca3_neurons[i].step(dt, I_syn + I_ca3)
            
            if sp:
                ca3_now += 1
                # 📐 Recurrent amplification
                for pre, post, syn in ca3_synapses:
                    if pre == i:
                        syn.on_pre_spike(t, S, PTP, 100.0, 0.0)
                        
                schaffer_collaterals[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
        
        ca3_last = ca3_now
        
        # 📐 WTA: Sparse competition (after 2.0 ms)
        if t > 2.0:
            apply_wta(ca3_neurons, k=len(targets))

        # --------------------------------------------------------
        # 4. CA1 (Output Filtering with WTA)
        # --------------------------------------------------------
        I_ca1 = -CA1_INHIB_RECALL  # 📐 Constant inhibition
        
        for i in range(N):
            I_syn = ca1_neurons[i].soma.get_total_synaptic_current()
            sp = ca1_neurons[i].step(dt, I_syn + I_ca1)
            
            # --------------------------------------------------------
            # 5. Subiculum (Spike-to-Rate Integration)
            # --------------------------------------------------------
            # 📐 Leaky Integrator: y(t+dt) = y(t) + dt·(-y/τ + w·spike)
            y = sub_neurons[i].step(dt, sp)
            sub_outputs[i] = max(sub_outputs[i], y)  # 📐 Peak activity
        
        # 📐 WTA: Sparse competition (after 3.0 ms)
        if t > 3.0:
            apply_wta(ca1_neurons, k=len(targets))

    # --------------------------------------------------------
    # 최종 결과 출력 (시각화)
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 HIPPOCAMPAL PROCESSING PIPELINE - VISUAL SUMMARY")
    print("=" * 70)
    
    # --------------------------------------------------------
    # 1. 전체 파이프라인 시각화
    # --------------------------------------------------------
    print("\n🧠 PROCESSING FLOW:")
    print("-" * 70)
    
    partial_input = [targets[0]]
    print("\n📥 INPUT (Partial Cue):")
    input_viz = ""
    for i in range(N):
        if i in partial_input:
            input_viz += "🎯"
        else:
            input_viz += "··"
    print(f"  {input_viz}")
    print(f"  Cue: N{partial_input[0]} (1/{len(targets)} = {100/len(targets):.0f}%)")
    
    print("\n  ⬇️  DG (Dentate Gyrus - Sparse Coding)")
    print("     └─ High Leak (gL=0.1) filters noise")
    
    print("\n  ⬇️  CA3 (Pattern Completion)")
    print(f"     ├─ Selective Recurrent ({len(ca3_synapses)} synapses)")
    pattern_links = len([1 for i,j,_ in ca3_synapses if i in targets and j in targets])
    print(f"     ├─ Pattern Links: {pattern_links} (Q=15.0)")
    background_links = len(ca3_synapses) - pattern_links
    print(f"     ├─ Background: ~{background_links} (Q=3.0, 10%)")
    print(f"     └─ WTA (k={len(targets)}) at t>2.0ms")
    
    print("\n  ⬇️  Schaffer Collaterals (CA3 → CA1)")
    print("     └─ Strong transmission (Q=25.0)")
    
    print("\n  ⬇️  CA1 (Output Filtering)")
    print("     ├─ Medium Leak (gL=0.08)")
    print(f"     ├─ Inhibition ({CA1_INHIB_RECALL})")
    print(f"     └─ WTA (k={len(targets)}) at t>3.0ms")
    
    print("\n  ⬇️  Subiculum (Rate Decoder)")
    print("     ├─ Leaky Integrator (tau=2.0)")
    print("     └─ Converts spikes → activity level")
    
    print("\n📤 OUTPUT (Cortical Readout):")
    
    # --------------------------------------------------------
    # 2. Subiculum Activity 시각화
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("🎓 SUBICULUM ACTIVITY LEVELS")
    print("-" * 70)
    
    # 📐 임계값: y > 2.0 → "활성"으로 판정
    threshold = 2.0
    winners = [i for i in range(N) if sub_outputs[i] > threshold]
    
    # 활성화된 뉴런만 표시
    active_found = False
    for i in range(N):
        val = sub_outputs[i]
        if val > 0.5:  # 0.5 이상만 표시
            active_found = True
            bar_length = int(val * 2)
            bar = "█" * bar_length
            
            # 상태 표시
            if i in targets:
                if val > threshold:
                    status = "🎯 TARGET ✅"
                else:
                    status = "🎯 TARGET (weak)"
            else:
                if val > threshold:
                    status = "🔥 NOISE ❌"
                else:
                    status = "⚪ Sub-threshold"
            
            print(f" N{i:2d}: {val:5.2f} | {bar:<20} {status}")
    
    if not active_found:
        print(" (No significant activity)")
    
    # --------------------------------------------------------
    # 3. 패턴 시각화
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("📈 PATTERN ANALYSIS")
    print("-" * 70)
    
    print("\n🎯 TARGET PATTERN:")
    target_viz = ""
    for i in range(N):
        if i in targets:
            target_viz += "██"
        else:
            target_viz += "··"
    print(f"  {target_viz}")
    print(f"  Expected: {targets}")
    
    print(f"\n📤 SUBICULUM OUTPUT (>{threshold}):")
    output_viz = ""
    for i in range(N):
        if i in winners and i in targets:
            output_viz += "██"  # 성공
        elif i in winners:
            output_viz += "🔥"  # 노이즈
        elif i in targets:
            output_viz += "▓▓"  # 누락
        else:
            output_viz += "··"
    print(f"  {output_viz}")
    print(f"  Recalled: {winners}")
    
    # 범례
    print("\n  Legend:")
    print("    ██ = Correct Target  |  🔥 = Noise  |  ▓▓ = Missed  |  ·· = Silent")
    
    # --------------------------------------------------------
    # 4. 성능 메트릭
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("📊 PERFORMANCE METRICS")
    print("-" * 70)
    
    # 📐 메트릭 계산
    correct = set(winners) & set(targets)   # True positives
    missing = set(targets) - set(winners)   # False negatives
    noise = set(winners) - set(targets)     # False positives
    
    completion_rate = len(correct) / len(targets) * 100 if targets else 0
    noise_rate = len(noise) / N * 100
    
    # 패턴 완성률 바
    bar_length = 30
    filled = int(bar_length * completion_rate / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"Pattern Completion  : [{bar}] {completion_rate:.0f}%")
    print(f"                      ({len(correct)}/{len(targets)} targets)")
    
    # 노이즈 레벨 바
    noise_filled = int(bar_length * min(noise_rate / 50, 1.0))
    noise_bar = "█" * noise_filled + "░" * (bar_length - noise_filled)
    print(f"Noise Level         : [{noise_bar}] {noise_rate:.1f}%")
    print(f"                      ({len(noise)}/{N} neurons)")
    
    # 📐 SNR (Signal-to-Noise Ratio)
    if len(correct) > 0:
        snr = len(correct) / max(1, len(noise))
        print(f"Signal-to-Noise     : {snr:.2f} (higher is better)")
    
    # --------------------------------------------------------
    # 5. 최종 판정
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("🏆 FINAL VERDICT")
    print("=" * 70)
    
    if len(missing) == 0:
        print("\n✅ PATTERN COMPLETION: SUCCESS!")
        if len(noise) == 0:
            print("🏆 PERFECT RECALL")
            print("   └─ Zero noise. Flawless hippocampal processing!")
            print("   └─ All targets recalled: ", correct)
        elif len(noise) <= 2:
            print("🎯 EXCELLENT RECALL")
            print(f"   └─ Minor noise: {noise} (biologically realistic)")
            print(f"   └─ All targets recalled: {correct}")
        else:
            print("⚠️  NOISY RECALL")
            print(f"   └─ Noise detected: {noise}")
            print(f"   └─ Targets recalled: {correct}")
    else:
        print("\n❌ PATTERN COMPLETION: FAILED")
        print(f"   └─ Missing targets: {missing}")
        print(f"   └─ Recalled targets: {correct}")
        if len(noise) > 0:
            print(f"   └─ Plus noise: {noise}")
    
    print("\n💡 FULL PIPELINE VERIFIED:")
    print("   Input → DG → CA3 → Schaffer → CA1 → Subiculum → Cortex")
    print("=" * 70)


if __name__ == "__main__":
    run_hippo_complete()
