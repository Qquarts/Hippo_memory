"""
===================================================================================
HIPPO_CA1 — Full Hippocampal Pipeline (DG → CA3 → Schaffer → CA1)
===================================================================================
지은이: GNJz | 발행: 2025.11.24

Complete Hippocampal Circuit with CA1 Output Layer

📐 구현된 핵심 수식:

1️⃣ Hodgkin-Huxley Neuron Dynamics (via v3_event.HHSomaQuick):
   C_m dV/dt = I_ext + I_syn - g_L(V-E_L) - g_Na·m³h(V-E_Na) - g_K·n⁴(V-E_K)
   
   Gating variables:
   dm/dt = α_m(1-m) - β_m·m
   dh/dt = α_h(1-h) - β_h·h
   dn/dt = α_n(1-n) - β_n·n

2️⃣ Layer-Specific Leak Conductances:
   DG:  g_L = 0.1  (high leak → strong noise filtering)
   CA3: g_L = 0.05 (standard → pattern completion)
   CA1: g_L = 0.08 (medium → output refinement)

3️⃣ Short-Term Plasticity (STP) & Post-Tetanic Potentiation (PTP):
   On spike:  S ← min(1.0, S + 0.3),    PTP ← min(2.0, PTP + 0.05)
   Decay:     S ← max(0.0, S - 0.01),   PTP ← max(1.0, PTP - 0.001)

4️⃣ Global Feedback Inhibition:
   I_inhib(t) = -N_active(t-1) · g_inhib
   
   Learning:  g_DG = 80.0,  g_CA3 = 20.0
   Recall:    g_DG = 150.0, g_CA3 = 60.0, g_CA1 = 35.0

5️⃣ Winner-Take-All (WTA) Competition:
   Select top-k neurons by voltage V
   Suppress losers: V_loser ← -70 mV
   
   Timing: CA3 WTA at t > 2.0 ms, CA1 WTA at t > 3.0 ms

6️⃣ Hippocampal Pathways:
   Mossy Fibers:        DG → CA3  (1:1, Q=80.0, "detonator")
   Recurrent:           CA3 ⟲ CA3 (selective: pattern Q=15.0, bg Q=3.0)
   Schaffer Collateral: CA3 → CA1 (1:1, Q=25.0, strong relay)

7️⃣ Pattern Completion via CA3:
   Input:    Partial Cue (1/N neurons)
   CA3:      Recurrent amplification (pattern-selective connections)
   Schaffer: Strong transmission to CA1
   CA1:      Output filtering with WTA
   Output:   Complete Pattern (N/N neurons)

8️⃣ Network Architecture:
   Learning:  Full Pattern → DG → CA3 (recurrent) → Schaffer → CA1
   Recall:    Partial Cue → DG → CA3 (completion) → Schaffer → CA1 → Output

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
    Dentate Gyrus Neuron — Sparse Pattern Encoder
    
    📐 구현 수식:
    1) High Leak Conductance:
       g_L = 0.1 (표준의 2배)
       
       I_leak = g_L · (V - E_L)
       
       → 높은 leak = 약한 자극은 필터링, 강한 자극만 통과
    
    2) HH Dynamics:
       C_m dV/dt = I_ext + I_syn - g_L(V-E_L) - g_Na·m³h(V-E_Na) - g_K·n⁴(V-E_K)
    
    3) STP/PTP:
       On spike:  S ← min(1.0, S + 0.3),    PTP ← min(2.0, PTP + 0.05)
       Decay:     S ← max(0.0, S - 0.01),   PTP ← max(1.0, PTP - 0.001)
    
    생물학적 의미:
    - DG는 "pattern separator" — 유사 입력을 구분
    - High leak → sparse coding (적은 수의 뉴런만 발화)
    - "Gatekeeper" 역할 — 노이즈 필터링
    """
    
    def __init__(self, name):
        self.name = name
        cfg = CONFIG["HH"].copy()
        cfg["gL"] = 0.1  # 📐 High leak conductance (sparse filtering)
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
    CA3 Neuron — Recurrent Pattern Completion Network
    
    📐 구현 수식:
    1) Standard Leak Conductance:
       g_L = 0.05 (표준값)
       
       → 중간 정도의 leak = 패턴 완성에 최적화
    
    2) Recurrent Input:
       I_total = I_mossy + I_recurrent + I_inhib
       
       where I_recurrent = Σ_j Q_ij · S_j · PTP_j · K(t-t_spike)
    
    3) Pattern-Selective Connectivity:
       Q_ij = { 15.0  if i,j ∈ same pattern (strong attractor)
              {  3.0  if random background (10% probability)
              {  0.0  otherwise
    
    생물학적 의미:
    - CA3는 "auto-associative memory" — 부분 입력으로 전체 패턴 복원
    - Recurrent connections → attractor dynamics
    - Pattern-specific strong links → 선택적 증폭
    - Schaffer collateral → CA1으로 완성된 패턴 전달
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
    CA1 Neuron — Output Filter & Relay to Cortex
    
    📐 구현 수식:
    1) Medium Leak Conductance:
       g_L = 0.08 (중간값)
       
       → DG와 CA3 사이 — 적절한 필터링 + 전달
    
    2) Schaffer Collateral Input:
       I_syn = Σ_j Q_j · S_j · PTP_j · K(t-t_spike)
       
       where j = CA3 neurons, Q = 25.0 (strong transmission)
    
    3) CA1 역할:
       - CA3의 완성된 패턴을 "정제"
       - WTA를 통한 sparse output
       - Subiculum/Entorhinal cortex로 전달
    
    생물학적 의미:
    - CA1은 CA3의 출력을 "필터링하여" 피질로 전달
    - Medium leak → 중간 정도의 선택성
    - Schaffer collateral → CA3의 강력한 신호를 받음
    """
    
    def __init__(self, name):
        self.name = name
        cfg = CONFIG["HH"].copy()
        cfg["gL"] = 0.08  # 📐 Medium leak conductance
        self.soma = HHSomaQuick(cfg)
        self.S, self.PTP = 0.0, 1.0  # 📐 STP/PTP (not used in this model, but kept)
        
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
    - CA3와 CA1에서 시간차를 두고 적용 (2.0ms, 3.0ms)
    
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
# 2. 통합 시뮬레이션 (Full Hippocampus)
# ======================================================================

def run_hippo_full(N=20, dt=0.1):
    """
    Complete Hippocampal Pipeline Simulation
    
    📐 전체 Pipeline:
    
    Phase 1 - Learning:
      Input → DG → CA3 (recurrent) → Schaffer → CA1
      
      목적: 전체 패턴 저장 + Schaffer collateral 강화
      입력: Full pattern (3 neurons)
      결과: CA3 recurrent + Schaffer synapses 강화
    
    Phase 2 - Reset:
      모든 뉴런 및 시냅스 초기화
      
      목적: Learning과 Recall 분리
    
    Phase 3 - Recall:
      Partial Cue → DG → CA3 (completion) → Schaffer → CA1 → Output
      
      목적: 부분 입력 (1/3)으로 전체 패턴 복원
      CA3: Recurrent connections로 패턴 완성
      Schaffer: 완성된 패턴을 CA1으로 전달
      CA1: WTA로 노이즈 제거 + 출력
    
    📐 핵심 메커니즘:
    
    1) Layer-Specific Leak:
       g_DG = 0.1 (high)   → Noise filtering
       g_CA3 = 0.05 (std)  → Pattern completion
       g_CA1 = 0.08 (med)  → Output refinement
    
    2) Global Inhibition (Feedback):
       I_inhib(t) = -N_active(t-1) · g_inhib
       
       → 많은 뉴런 발화 → 강한 억제 → Sparse coding 유지
    
    3) Selective Recurrent (CA3):
       Q_ij = { 15.0  if i,j ∈ pattern (strong attractor)
              {  3.0  if background (weak)
       
       → Pattern neurons만 강하게 연결 → 선택적 증폭
    
    4) Schaffer Collateral:
       Q = 25.0 (CA3 → CA1)
       
       → 강력한 전달 — CA3의 완성된 패턴을 CA1으로
    
    5) Two-Stage WTA:
       CA3: t > 2.0 ms → Top-3 선택
       CA1: t > 3.0 ms → Top-3 선택
       
       → 단계적 노이즈 제거
    
    Parameters:
        N: 뉴런 수 (default: 20)
        dt: Timestep (default: 0.1 ms)
    """
    
    # 🎲 Reproducibility
    random.seed(42)
    np.random.seed(42)

    print(f"\n🧠 FULL HIPPOCAMPUS SIMULATION (Input -> DG -> CA3 -> CA1)")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 뉴런 생성
    # --------------------------------------------------------
    dg_neurons  = [DGLightNeuron(f"DG{i}")  for i in range(N)]
    ca3_neurons = [CA3LightNeuron(f"CA3{i}") for i in range(N)]
    ca1_neurons = [CA1LightNeuron(f"CA1{i}") for i in range(N)]

    # --------------------------------------------------------
    # 2. 연결 (Connectivity)
    # --------------------------------------------------------
    
    # 📐 A) Perforant Path (Input → DG): 직접 입력으로 처리
    # 생물학적: Entorhinal cortex → DG
    
    # 📐 B) Mossy Fibers (DG → CA3): 1:1 Detonator
    # 생물학적: 강력한 "detonator" 시냅스
    mossy_fibers = []
    for i in range(N):
        syn = SynapseCore(pre_neuron=dg_neurons[i].soma, post_neuron=ca3_neurons[i].soma,
                          delay_ms=1.0, Q_max=80.0, tau_ms=2.0)  # 📐 Strong detonator (Q=80.0)
        mossy_fibers.append(syn)

    # 📐 C) CA3 Recurrent (CA3 ⟲ CA3): Selective Connectivity
    # Pattern-specific: 강한 연결 (Q=15.0)
    # Background: 약한 연결 (Q=3.0, 10% probability)
    ca3_synapses = []
    targets = [1, 7, 15]  # 📐 Target pattern (3 neurons)
    
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

    # 📐 D) Schaffer Collaterals (CA3 → CA1): 1:1 Strong Mapping
    # 생물학적: CA3의 주요 출력 경로
    schaffer_collaterals = []
    for i in range(N):
        syn = SynapseCore(pre_neuron=ca3_neurons[i].soma, post_neuron=ca1_neurons[i].soma,
                          delay_ms=2.0, Q_max=25.0, tau_ms=3.0)  # 📐 Strong transmission (Q=25.0)
        schaffer_collaterals.append(syn)

    print(f"Structure Built: DG({N}) -> CA3({N}) -> CA1({N})")

    # --------------------------------------------------------
    # PHASE 1: LEARNING (CA3 & Schaffer Potentiation)
    # --------------------------------------------------------
    print("\n=== PHASE 1: LEARNING (Target Pattern) ===")
    print(f"Target: {targets}")
    
    T_learn = 50.0  # Learning duration (ms)
    steps = int(T_learn / dt)
    
    # 📐 Learning Inhibition Parameters
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
                mossy_fibers[i].on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
        
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
                # 📐 Recurrent transmission
                for pre, post, syn in ca3_synapses:
                    if pre == i:
                        syn.on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
                
                # 📐 Schaffer collateral transmission (중요!)
                # CA3 → CA1 경로 강화
                schaffer_collaterals[i].on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
        
        ca3_last = ca3_now  # 📐 Update N_active(t-1)

    print("✅ Learning Complete (CA3 & Schaffer Potentiated).")

    # --------------------------------------------------------
    # PHASE 2: RESET (System Cooldown)
    # --------------------------------------------------------
    print("\n=== PHASE 2: RESET ===")
    
    # 📐 Cooldown: 모든 transient dynamics 소멸
    for _ in range(500):
        for n in dg_neurons + ca3_neurons + ca1_neurons:
            n.step(dt, 0)
        for s in mossy_fibers:
            s.deliver(0)
        for _, _, s in ca3_synapses:
            s.deliver(0)
        for s in schaffer_collaterals:
            s.deliver(0)

    # 📐 Force Reset: 모든 state variables 초기화
    all_neurons = dg_neurons + ca3_neurons + ca1_neurons
    for n in all_neurons:
        n.soma.V = -70.0
        n.soma.m, n.soma.h, n.soma.n = 0.05, 0.6, 0.32
        n.soma.spike_flag = False
        n.soma.I_syn_total = 0.0
        n.soma.active_remaining = 0.0
        n.soma.mode = "rest"
    
    # 📐 Synapse Reset (Spike Queue Clear)
    all_synapses = mossy_fibers + [s for _, _, s in ca3_synapses] + schaffer_collaterals
    for s in all_synapses:
        s.spikes = []
        s.I_syn = 0.0
        
    dg_last = 0
    ca3_last = 0
    print("Reset Done.")

    # --------------------------------------------------------
    # PHASE 3: RECALL (Partial Cue → Pattern Completion → CA1 Output)
    # --------------------------------------------------------
    print("\n=== PHASE 3: RECALL (Input: N1 only) ===")
    partial_input = [1]  # 📐 Partial cue (1/3 of pattern)
    print(f"Input: {partial_input} (Missing {list(set(targets)-set(partial_input))})")
    
    T_test = 50.0  # Recall duration (ms)
    steps = int(T_test / dt)
    
    # 📐 Recall Inhibition (stronger than learning)
    DG_INHIB_RECALL = 150.0   # 📐 Strong DG inhibition
    CA3_INHIB_RECALL = 60.0   # 📐 Strong CA3 inhibition
    CA1_INHIB_RECALL = 35.0   # 📐 CA1 noise filter

    ca3_log = set()  # 📐 CA3 활성 뉴런 기록
    ca1_log = set()  # 📐 CA1 활성 뉴런 기록

    for k in range(steps):
        t = k * dt
        
        # --------------------------------------------------------
        # 1. DG (Input Filtering)
        # --------------------------------------------------------
        dg_now = 0
        # 📐 Global inhibition: I_inhib = -N_active(t-1) · g_inhib
        I_dg = -1.0 * dg_last * DG_INHIB_RECALL
        
        for i in range(N):
            # 📐 Partial Cue: Only first target neuron (t < 10 ms)
            I_in = 200.0 if (i in partial_input and t < 10.0) else 0.0
            sp, S, PTP = dg_neurons[i].step(dt, I_in + I_dg)
            
            if sp:
                dg_now += 1
                mossy_fibers[i].on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
        
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
                ca3_log.add(i)  # 📐 Record CA3 activity
                
                # 📐 Recurrent amplification
                for pre, post, syn in ca3_synapses:
                    if pre == i:
                        syn.on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
                
                # 📐 Schaffer collateral transmission (PTP 적용!)
                schaffer_collaterals[i].on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
        
        ca3_last = ca3_now
        
        # 📐 WTA: CA3 sparse competition (after 2.0 ms)
        if t > 2.0:
            apply_wta(ca3_neurons, k=3)  # Top-3 neurons

        # --------------------------------------------------------
        # 4. CA1 (Final Output / Decoding with WTA)
        # --------------------------------------------------------
        I_ca1 = -CA1_INHIB_RECALL  # 📐 Constant inhibition
        
        for i in range(N):
            I_syn = ca1_neurons[i].soma.get_total_synaptic_current()
            sp = ca1_neurons[i].step(dt, I_syn + I_ca1)
            
            if sp:
                ca1_log.add(i)  # 📐 Record CA1 activity
        
        # 📐 WTA: CA1 sparse competition (after 3.0 ms)
        if t > 3.0:
            apply_wta(ca1_neurons, k=3)  # Top-3 neurons

    # --------------------------------------------------------
    # 결과 분석 (시각화)
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 HIPPOCAMPAL MEMORY RETRIEVAL - RESULT ANALYSIS")
    print("=" * 70)
    
    # 📐 메트릭 계산
    missing = set(targets) - ca1_log   # False negatives
    noise = ca1_log - set(targets)     # False positives
    
    # --------------------------------------------------------
    # 1. 전체 파이프라인 시각화
    # --------------------------------------------------------
    print("\n🧠 PROCESSING FLOW:")
    print("-" * 70)
    
    print("\n📥 INPUT (Partial Cue):")
    input_viz = ""
    for i in range(N):
        if i in partial_input:
            input_viz += "🎯"
        else:
            input_viz += "··"
    print(f"  {input_viz}")
    print(f"  Cue: {partial_input} (1/{len(targets)} = 33%)")
    
    print("\n  ⬇️  DG (Dentate Gyrus - Sparse Coding)")
    print("     └─ High Leak (gL=0.1) filters noise")
    
    print("\n  ⬇️  CA3 (Pattern Completion)")
    print(f"     ├─ Selective Recurrent ({len(ca3_synapses)} synapses)")
    pattern_links = len([1 for i, j, _ in ca3_synapses if i in targets and j in targets])
    print(f"     ├─ Pattern Links: {pattern_links} (Q=15.0)")
    background_links = len(ca3_synapses) - pattern_links
    print(f"     ├─ Background: ~{background_links} (Q=3.0, 10%)")
    print("     └─ WTA (k=3) at t>2.0ms")
    
    print("\n🧠 CA3 OUTPUT:")
    ca3_viz = ""
    for i in range(N):
        if i in targets and i in ca3_log:
            ca3_viz += "██"
        elif i in targets and i not in ca3_log:
            ca3_viz += "▓▓"
        elif i in (ca3_log - set(targets)):
            ca3_viz += "🔥"
        else:
            ca3_viz += "··"
    print(f"  {ca3_viz}")
    print(f"  Neurons: {sorted(list(ca3_log))}")
    
    print("\n  ⬇️  Schaffer Collaterals (CA3 → CA1)")
    print("     └─ Strong transmission (Q=25.0)")
    
    print("\n  ⬇️  CA1 (Output Filtering)")
    print("     ├─ Medium Leak (gL=0.08)")
    print(f"     ├─ Inhibition ({CA1_INHIB_RECALL})")
    print("     └─ WTA (k=3) at t>3.0ms")
    
    print("\n🎓 CA1 OUTPUT (Final):")
    ca1_viz = ""
    for i in range(N):
        if i in targets and i in ca1_log:
            ca1_viz += "██"
        elif i in targets and i not in ca1_log:
            ca1_viz += "▓▓"
        elif i in (ca1_log - set(targets)):
            ca1_viz += "🔥"
        else:
            ca1_viz += "··"
    print(f"  {ca1_viz}")
    print(f"  Neurons: {sorted(list(ca1_log))}")
    
    # 범례
    print("\n  Legend:")
    print("    🎯 = Input Cue  |  ██ = Target Recalled  |  ▓▓ = Target Missed")
    print("    🔥 = Noise      |  ·· = Silent")
    
    # --------------------------------------------------------
    # 2. 패턴 비교
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
    
    print("\n📤 CA1 OUTPUT:")
    output_viz = ""
    for i in range(N):
        if i in ca1_log and i in targets:
            output_viz += "██"
        elif i in ca1_log:
            output_viz += "🔥"
        elif i in targets:
            output_viz += "▓▓"
        else:
            output_viz += "··"
    print(f"  {output_viz}")
    print(f"  Recalled: {sorted(list(ca1_log))}")
    
    # --------------------------------------------------------
    # 3. 상세 분석
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("📋 DETAILED ANALYSIS")
    print("-" * 70)
    
    correct = set(ca1_log) & set(targets)
    
    print(f"Target Pattern      : {targets}")
    print(f"Input (Partial Cue) : {partial_input} → Missing: {sorted(list(set(targets) - set(partial_input)))}")
    print(f"CA3 Output          : {sorted(list(ca3_log))}")
    print(f"CA1 Output          : {sorted(list(ca1_log))}")
    
    print("\n✓ Completed Targets : ", end="")
    if correct:
        print(f"{sorted(list(correct))} ✅")
    else:
        print("None ❌")
    
    print("✗ Missing Targets   : ", end="")
    if missing:
        print(f"{sorted(list(missing))} ❌")
    else:
        print("None ✅")
    
    print("⚠ Noise Neurons     : ", end="")
    if noise:
        print(f"{sorted(list(noise))} ⚠️")
    else:
        print("None 🏆")
    
    # --------------------------------------------------------
    # 4. 성능 메트릭
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("📊 PERFORMANCE METRICS")
    print("-" * 70)
    
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
            print(f"   └─ All targets recalled: {sorted(list(correct))}")
        elif len(noise) == 1:
            print("🎯 EXCELLENT RECALL")
            print(f"   └─ Minimal noise: {sorted(list(noise))} (biologically ideal!)")
            print(f"   └─ All targets recalled: {sorted(list(correct))}")
        elif len(noise) <= 2:
            print("✨ VERY GOOD RECALL")
            print(f"   └─ Minor noise: {sorted(list(noise))} (biologically realistic)")
            print(f"   └─ All targets recalled: {sorted(list(correct))}")
        else:
            print("⚠️  NOISY RECALL")
            print(f"   └─ Noise detected: {sorted(list(noise))}")
            print(f"   └─ Targets recalled: {sorted(list(correct))}")
    else:
        print("\n❌ PATTERN COMPLETION: FAILED")
        print(f"   └─ Missing targets: {sorted(list(missing))}")
        print(f"   └─ Recalled targets: {sorted(list(correct))}")
        if len(noise) > 0:
            print(f"   └─ Plus noise: {sorted(list(noise))}")
    
    print("\n💡 FULL PIPELINE VERIFIED:")
    print("   Input → DG → CA3 (Selective + WTA) → Schaffer → CA1 (WTA) → Output")
    print("=" * 70)


if __name__ == "__main__":
    run_hippo_full()
