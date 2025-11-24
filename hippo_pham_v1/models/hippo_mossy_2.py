"""
===================================================================================
HIPPO_MOSSY_2 — DG→CA3 with Selective Recurrent + WTA
===================================================================================
지은이: GNJz | 발행: 2025.11.24

Hippocampal Circuit: DG → CA3 (Selective Recurrent Network) with WTA

⚠️ Note: This is a DG→CA3 selective recurrent test module.
   CA1 and Schaffer collaterals are NOT included (see HIPPO_CA1.py for full pipeline).

📐 구현된 핵심 수식:

1️⃣ Hodgkin-Huxley Neuron Dynamics (via v3_event.HHSomaQuick):
   C_m dV/dt = I_ext + I_syn - g_L(V-E_L) - g_Na·m³h(V-E_Na) - g_K·n⁴(V-E_K)
   
   DG:  g_L = 0.1  (high leak → noise filtering)
   CA3: g_L = 0.05 (standard → pattern completion)

2️⃣ Short-Term Plasticity (STP) & Post-Tetanic Potentiation (PTP):
   On spike:  S ← min(1.0, S + 0.3),    PTP ← min(2.0, PTP + 0.05)
   Decay:     S ← max(0.0, S - 0.01),   PTP ← max(1.0, PTP - 0.001)

3️⃣ Global Feedback Inhibition:
   I_inhib(t) = -N_active(t-1) · g_inhib
   
   Learning: g_DG = 80.0,  g_CA3 = 20.0
   Recall:   g_DG = 150.0, g_CA3 = 60.0

4️⃣ Selective Recurrent Connectivity (CA3):
   Q_ij = { 15.0  if i,j ∈ pattern (strong attractor, 100% connection)
          {  3.0  if background (weak, 10% probability)
          {  0.0  otherwise
   
   Pattern: [N1, N7, N15] → 6 strong links (3×2)
   Background: ~40 weak links (random 10%)

5️⃣ Winner-Take-All (WTA) — Aggressive Suppression:
   1) Sort neurons by voltage: V_sorted = sort([V_1, ..., V_N], desc)
   2) Select top-k: Winners = {i | V_i ∈ top-k}
   3) Suppress losers: V_loser ← -70 mV (forced reset)
   
   📐 Timing: CA3 WTA applied at t > 2.0 ms (k=5)
   → After mossy fiber + initial recurrent activation
   → Maintains top-5 neurons, suppresses all others

6️⃣ Pattern Completion Test:
   Input:  Partial Cue (1/3 neurons) → [N1]
   CA3:    Recurrent amplification
   Output: Complete Pattern (3/3 neurons) → [N1, N7, N15]

7️⃣ Hippocampal Pathways:
   Mossy Fibers: DG → CA3 (1:1, Q=80.0, "detonator")
   Recurrent:    CA3 ⟲ CA3 (selective: pattern Q=15.0, bg Q=3.0)

8️⃣ Network Architecture:
   Learning: Full Pattern → DG → CA3 (recurrent strengthens)
   Recall:   Partial Cue → DG → CA3 (pattern completion + WTA)

===================================================================================
"""

# Qquarts co Present
# 지은이 : GNJz 
# 발행 2025.11.24

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
# 1. 뉴런 클래스 (동일)
# ======================================================================

class DGLightNeuron:
    """
    Dentate Gyrus Neuron — Sparse Pattern Encoder
    
    📐 High Leak (g_L = 0.1) → Noise filtering
    
    수식: hippo_sub.py의 DGLightNeuron과 동일
    """
    def __init__(self, name):
        self.name = name
        cfg = CONFIG["HH"].copy()
        cfg["gL"] = 0.1  # 📐 High leak conductance
        self.soma = HHSomaQuick(cfg)
        self.S, self.PTP, self.dphi = 0.0, 1.0, 0.0

    def step(self, dt, I_ext=0.0):
        self.soma.step(dt, I_ext=I_ext)
        Vm = self.soma.V
        spike = self.soma.spiking()
        
        # 📐 STP/PTP update
        if spike:
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
            
        return Vm, spike, self.S, self.PTP, self.dphi


class CA3LightNeuron:
    """
    CA3 Neuron — Recurrent Attractor Network
    
    📐 Standard Leak (g_L = 0.05) → Pattern completion
    
    수식: hippo_sub.py의 CA3LightNeuron과 동일
    """
    def __init__(self, name):
        self.name = name
        self.soma = HHSomaQuick(CONFIG["HH"]) 
        self.S, self.PTP, self.dphi = 0.0, 1.0, 0.0

    def step(self, dt, I_ext=0.0):
        self.soma.step(dt, I_ext=I_ext)
        Vm = self.soma.V
        spike = self.soma.spiking()
        
        # 📐 STP/PTP update
        if spike:
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
            
        return Vm, spike, self.S, self.PTP, self.dphi


# ======================================================================
# 2. WTA Helper (승자 독식)
# ======================================================================

def apply_ca3_wta(neurons, k=5):
    """
    Winner-Take-All (WTA) — Sparse Competition with Aggressive Suppression
    
    📐 구현 수식:
    1) 전압 기준 정렬:
       V_sorted = sort([V_1, V_2, ..., V_N], descending)
    
    2) Top-k 선택:
       Winners = {i | V_i ∈ top-k}
    
    3) Losers 강제 억제:
       For i ∈ Losers:
         if V_i > -60 mV: V_i ← -70 mV (forced reset)
    
    📐 핵심 차이 (vs. hippo_sub.py):
    - Winners list 반환 (디버깅/분석용)
    - "Aggressive suppression" — 발화 중인 뉴런도 강제로 억제
    
    생물학적 의미:
    - GABAergic interneuron의 strong feedback inhibition
    - Sparse coding 유지 (CA3에서 소수만 활성화)
    - 노이즈 뉴런 강제 억제
    
    Parameters:
        neurons: 뉴런 리스트
        k: 승자 수 (default: 5)
    
    Returns:
        winners: 승자 뉴런 인덱스 리스트
    """
    # 📐 수식: V_sorted = sort([V_1, ..., V_N], descending)
    voltages = [(i, n.soma.V) for i, n in enumerate(neurons)]
    voltages.sort(key=lambda x: x[1], reverse=True)
    
    # 📐 수식: Winners = {i | V_i ∈ top-k}
    winners = [idx for idx, _ in voltages[:k]]
    losers = [idx for idx, _ in voltages[k:]]
    
    # 📐 수식: V_loser ← -70 mV (aggressive suppression)
    for idx in losers:
        # 발화 중인 뉴런도 강제로 억제 (aggressive)
        if neurons[idx].soma.V > -60.0:
            neurons[idx].soma.V = -70.0
            neurons[idx].soma.spike_flag = False
            neurons[idx].soma.mode = "rest"
    
    return winners


# ======================================================================
# 3. 통합 시뮬레이션 (Selective Recurrent)
# ======================================================================

def run_hippo_final_v2(N=20, dt=0.1):
    """
    Hippocampus DG→CA3 with Selective Recurrent + WTA
    
    📐 핵심 메커니즘:
    
    1) Selective Recurrent Connectivity:
       - Pattern neurons (N1, N7, N15): 100% connected, Q=15.0
       - Background neurons: 10% connected, Q=3.0
       
       → Pattern neurons form strong attractor
       → Background provides weak baseline activity
    
    2) Winner-Take-All (WTA):
       - Applied at t > 2.0 ms
       - Top-5 neurons survive
       - Aggressive suppression of losers
       
       → Noise removal
       → Sparse representation
    
    3) Pattern Completion Test:
       Input:  [N1] (1/3 of pattern)
       CA3:    Recurrent amplification
       Output: [N1, N7, N15] (3/3 complete pattern)
    
    Parameters:
        N: 뉴런 수 (default: 20)
        dt: Timestep (default: 0.1 ms)
    """
    
    # 🎲 Reproducibility
    random.seed(42)
    np.random.seed(42)

    print(f"\n🧠 HIPPOCAMPUS FINAL V2 (Selective Recurrent + WTA)")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. 뉴런 생성
    # --------------------------------------------------------
    dg_neurons = [DGLightNeuron(f"DG{i}") for i in range(N)]
    ca3_neurons = [CA3LightNeuron(f"CA3{i}") for i in range(N)]
    targets = [1, 7, 15]  # 📐 Target pattern (3 neurons)

    # --------------------------------------------------------
    # 2. 연결 (Connectivity) - Selective
    # --------------------------------------------------------
    
    # 📐 Mossy Fibers (DG → CA3): 1:1 Detonator
    mossy_fibers = []
    for i in range(N):
        syn = SynapseCore(pre_neuron=dg_neurons[i].soma, 
                          post_neuron=ca3_neurons[i].soma,
                          delay_ms=1.0, Q_max=80.0, tau_ms=2.0)  # 📐 Strong detonator
        mossy_fibers.append(syn)

    # 📐 CA3 Recurrent: Selective Connectivity
    ca3_synapses = []
    pattern_links = 0
    background_links = 0
    
    for i in range(N):
        for j in range(N):
            if i == j: continue
            
            # 📐 Pattern-selective connectivity
            is_pattern_link = (i in targets) and (j in targets)
            
            if is_pattern_link:
                # 📐 Strong attractor: Q=15.0 (pattern neurons, 100% connected)
                syn = SynapseCore(pre_neuron=ca3_neurons[i].soma,
                                  post_neuron=ca3_neurons[j].soma,
                                  delay_ms=1.5, 
                                  Q_max=15.0,  # Strong
                                  tau_ms=3.0)
                ca3_synapses.append((i, j, syn))
                pattern_links += 1
                
            elif random.random() < 0.10: 
                # 📐 Weak background: Q=3.0 (10% random)
                syn = SynapseCore(pre_neuron=ca3_neurons[i].soma,
                                  post_neuron=ca3_neurons[j].soma,
                                  delay_ms=1.5, 
                                  Q_max=3.0,  # Weak
                                  tau_ms=3.0)
                ca3_synapses.append((i, j, syn))
                background_links += 1

    print(f"Structure: Selective Recurrent")
    print(f" - Pattern Links (Q=15.0) : {pattern_links}")
    print(f" - Background (Q=3.0)     : {background_links}")
    print(f" - Total CA3 Synapses     : {len(ca3_synapses)}")

    # --------------------------------------------------------
    # PHASE 1: LEARNING
    # --------------------------------------------------------
    print("\n=== LEARNING PHASE ===")
    T_learn = 50.0
    steps = int(T_learn / dt)
    
    # 📐 Learning Inhibition (moderate)
    DG_INHIB = 80.0
    CA3_INHIB = 20.0
    
    dg_last_active = 0
    ca3_last_active = 0

    for k in range(steps):
        t = k * dt
        
        # 📐 Input: Full pattern (t < 10 ms)
        dg_input = [0.0] * N
        if t < 10.0:
            for idx in targets:
                dg_input[idx] = 200.0  # 📐 Strong input
        
        # DG Step
        dg_active_now = 0
        I_inhib_dg = -1.0 * dg_last_active * DG_INHIB  # 📐 Global inhibition
        
        for i in range(N):
            Vm, sp, S, PTP, _ = dg_neurons[i].step(dt, I_ext=dg_input[i] + I_inhib_dg)
            if sp:
                dg_active_now += 1
                mossy_fibers[i].on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
        dg_last_active = dg_active_now

        # Synapse Deliver
        for syn in mossy_fibers:
            syn.deliver(t)
        for _, _, syn in ca3_synapses:
            syn.deliver(t)

        # CA3 Step
        ca3_active_now = 0
        I_inhib_ca3 = -1.0 * ca3_last_active * CA3_INHIB  # 📐 Global inhibition
        
        for i in range(N):
            I_mossy = ca3_neurons[i].soma.get_total_synaptic_current()
            Vm, sp, S, PTP, _ = ca3_neurons[i].step(dt, I_ext=I_mossy + I_inhib_ca3)
            
            if sp:
                ca3_active_now += 1
                # 📐 Recurrent transmission
                for pre, post, syn in ca3_synapses:
                    if pre == i:
                        syn.on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
        ca3_last_active = ca3_active_now

    print("✅ Learning Complete.")

    # --------------------------------------------------------
    # PHASE 2: RESET
    # --------------------------------------------------------
    print("\n=== RESET ===")
    
    # 📐 Cooldown
    for _ in range(500):
        for n in dg_neurons + ca3_neurons:
            n.step(dt, 0)
        for s in mossy_fibers:
            s.deliver(0)
        for _, _, s in ca3_synapses:
            s.deliver(0)
    
    # 📐 Force Reset
    for n in dg_neurons + ca3_neurons:
        n.soma.V = -70.0
        n.soma.m, n.soma.h, n.soma.n = 0.05, 0.6, 0.32
        n.soma.spike_flag = False
        n.soma.active_remaining = 0.0
        n.soma.mode = "rest"
        n.soma.I_syn_total = 0.0
        
    for s in mossy_fibers:
        s.spikes = []
        s.I_syn = 0.0
    for _, _, s in ca3_synapses:
        s.spikes = []
        s.I_syn = 0.0
    
    dg_last_active = 0
    ca3_last_active = 0
    print("Reset Done.")

    # --------------------------------------------------------
    # PHASE 3: RECALL (Partial Cue + WTA)
    # --------------------------------------------------------
    print("\n=== RECALL PHASE (Partial Cue + WTA) ===")
    partial_input = [1]  # 📐 Partial cue (1/3 of pattern)
    print(f"Input: Only {partial_input} (Missing: {list(set(targets)-set(partial_input))})")

    T_test = 50.0
    steps = int(T_test / dt)
    ca3_logs = []
    
    # 📐 Recall Inhibition (strong)
    DG_INHIB_RECALL = 150.0
    CA3_INHIB_RECALL = 60.0

    print("Running Simulation...")
    
    for k in range(steps):
        t = k * dt
        
        # DG Step (Partial Input - Clean)
        dg_active_now = 0
        I_inhib_dg = -1.0 * dg_last_active * DG_INHIB_RECALL  # 📐 Strong inhibition
        
        for i in range(N):
            # 📐 Partial cue (clean, no noise)
            I_in = 200.0 if (i in partial_input and t < 10.0) else 0.0
            
            Vm, sp, S, PTP, _ = dg_neurons[i].step(dt, I_ext=I_in + I_inhib_dg)
            if sp:
                dg_active_now += 1
                mossy_fibers[i].on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
        dg_last_active = dg_active_now

        # Synapse Deliver
        for syn in mossy_fibers:
            syn.deliver(t)
        for _, _, syn in ca3_synapses:
            syn.deliver(t)

        # CA3 Step (Pattern Completion)
        ca3_active_now = 0
        ca3_spikes = []
        
        I_inhib_ca3 = -1.0 * ca3_last_active * CA3_INHIB_RECALL  # 📐 Strong inhibition
        
        for i in range(N):
            I_syn = ca3_neurons[i].soma.get_total_synaptic_current()
            Vm, sp, S, PTP, _ = ca3_neurons[i].step(dt, I_ext=I_syn + I_inhib_ca3)
            
            if sp:
                ca3_active_now += 1
                ca3_spikes.append(i)
                # 📐 Recurrent amplification (during recall)
                for pre, post, syn in ca3_synapses:
                    if pre == i:
                        syn.on_pre_spike(t, S, PTP)  # 📐 Uses default: ATP=100.0, dphi=0.0
        
        ca3_last_active = ca3_active_now
        if ca3_spikes:
            ca3_logs.append((t, ca3_spikes))
        
        # 📐 WTA: Aggressive suppression (after 2.0 ms)
        if t > 2.0:
            apply_ca3_wta(ca3_neurons, k=5)  # Top-5만 생존

    # --------------------------------------------------------
    # 결과 분석 (시각화)
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 PATTERN COMPLETION TEST - RESULT ANALYSIS")
    print("=" * 70)
    
    # 📐 메트릭 계산
    recalled_set = set()
    for t, ids in ca3_logs:
        for i in ids:
            recalled_set.add(i)
    
    missing = set(targets) - recalled_set
    noise = recalled_set - set(targets)
    
    # --------------------------------------------------------
    # 1. INPUT/OUTPUT 시각화
    # --------------------------------------------------------
    print("\n🎯 INPUT PATTERN (Partial Cue):")
    input_viz = ""
    for i in range(N):
        if i in partial_input:
            input_viz += "🎯"
        else:
            input_viz += "··"
    print(f"  {input_viz}")
    print(f"  Neurons: {partial_input} (1/{len(targets)} = 33% cue)")
    
    print("\n🧠 OUTPUT PATTERN (CA3 Recall):")
    output_viz = ""
    for i in range(N):
        if i in targets and i in recalled_set:
            output_viz += "██"  # 성공
        elif i in targets and i not in recalled_set:
            output_viz += "▓▓"  # 누락
        elif i in noise:
            output_viz += "🔥"  # 노이즈
        else:
            output_viz += "··"
    print(f"  {output_viz}")
    
    # 범례
    print("\n  Legend:")
    print("    🎯 = Input Cue  |  ██ = Target Recalled  |  ▓▓ = Target Missed")
    print("    🔥 = Noise      |  ·· = Silent")
    
    # --------------------------------------------------------
    # 2. 상세 분석
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("📈 DETAILED ANALYSIS")
    print("-" * 70)
    
    print(f"Target Pattern      : {targets}")
    print(f"Input (Partial Cue) : {partial_input} → Missing: {sorted(list(set(targets) - set(partial_input)))}")
    print(f"CA3 Fired Neurons   : {sorted(list(recalled_set))}")
    
    print("\n✓ Completed Targets : ", end="")
    if recalled_set & set(targets):
        print(f"{sorted(list(recalled_set & set(targets)))} ✅")
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
    # 3. 성능 메트릭
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("📊 PERFORMANCE METRICS")
    print("-" * 70)
    
    completion_rate = len(recalled_set & set(targets)) / len(targets) * 100
    noise_rate = len(noise) / N * 100
    
    # 패턴 완성률 바
    bar_length = 30
    filled = int(bar_length * completion_rate / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"Pattern Completion  : [{bar}] {completion_rate:.0f}%")
    print(f"                      ({len(recalled_set & set(targets))}/{len(targets)} targets recalled)")
    
    # 노이즈 레벨 바
    noise_filled = int(bar_length * min(noise_rate / 50, 1.0))
    noise_bar = "█" * noise_filled + "░" * (bar_length - noise_filled)
    print(f"Noise Level         : [{noise_bar}] {noise_rate:.1f}%")
    print(f"                      ({len(noise)}/{N} neurons)")
    
    # 📐 SNR
    if len(recalled_set & set(targets)) > 0:
        snr = len(recalled_set & set(targets)) / max(1, len(noise))
        print(f"Signal-to-Noise     : {snr:.2f} (higher is better)")
    
    # --------------------------------------------------------
    # 4. 최종 평가
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("🏆 FINAL VERDICT")
    print("=" * 70)
    
    if len(missing) == 0:
        print("\n✅ PATTERN COMPLETION: SUCCESS!")
        if len(noise) == 0:
            print("🏆 PERFECT RECALL")
            print("   └─ Zero noise detected. Flawless pattern completion!")
        elif len(noise) <= 2:
            print("🎯 EXCELLENT RECALL")
            print(f"   └─ Minor noise detected ({len(noise)} neurons). Biologically realistic!")
        elif len(noise) <= 5:
            print("⚠️  GOOD RECALL")
            print(f"   └─ Moderate noise detected ({len(noise)} neurons). Acceptable performance.")
        else:
            print("❌ NOISY RECALL")
            print(f"   └─ High noise detected ({len(noise)} neurons). Needs improvement.")
    else:
        print("\n❌ PATTERN COMPLETION: FAILED")
        print(f"   └─ Missing {len(missing)} target(s): {sorted(list(missing))}")
        if len(noise) > 0:
            print(f"   └─ Plus {len(noise)} noise neuron(s): {sorted(list(noise))}")
    
    print("=" * 70)


if __name__ == "__main__":
    run_hippo_final_v2()
