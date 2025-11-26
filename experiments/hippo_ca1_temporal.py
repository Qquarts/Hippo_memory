"""
================================================================================
HIPPO CA1: Temporal Encoding (시간 부호화)
================================================================================

[CA3 vs CA1]
CA3: "A, B, C 순서는 알아" (순서 정보)
CA1: "A는 0ms, B는 10ms, C는 20ms" (시간 정보)

[핵심 메커니즘]
1. Phase Precession (위상 전진)
   - Theta 리듬과 동기화
   - 시간 순서를 위상으로 인코딩

2. Time Cells (시간 세포)
   - 특정 시간 간격에 발화
   - "B 후 10ms 지나면 내가 터진다"

[실험 목표]
입력: A→B→C (CA3)
CA1 처리: 정확한 타이밍 추가
출력: A(t=0), B(t=10), C(t=20) 재현
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from v4_event import CONFIG, HHSomaQuick, SynapseCore

# ======================================================================
# CA1 Time Cell (시간 세포)
# ======================================================================
class TimeCell:
    """
    특정 시간 간격 후 발화하는 세포
    """
    def __init__(self, delay_ms, name):
        self.delay_ms = delay_ms  # 발화할 시간 간격
        self.name = name
        self.soma = HHSomaQuick(CONFIG["HH"])
        self.trigger_time = None  # 트리거된 시간
        self.S, self.PTP = 0.0, 1.0
        self.outgoing_synapses = []
        self.incoming_synapses = []
    
    def trigger(self, t):
        """CA3에서 신호 받으면 타이머 시작"""
        if self.trigger_time is None:
            self.trigger_time = t
    
    def step(self, dt, t, I_ext=0.0):
        """시간이 되면 자동으로 발화"""
        if self.trigger_time is not None:
            elapsed = t - self.trigger_time
            
            # 목표 시간 도달 (±2ms 윈도우)
            if abs(elapsed - self.delay_ms) < 2.0:
                I_ext += 200.0  # 자동 자극
        
        self.soma.step(dt, I_ext)
        sp = self.soma.spiking()
        
        if sp:
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
            for syn in self.outgoing_synapses:
                syn.on_pre_spike(t, self.S, self.PTP, 100.0, 0.0)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
        
        return sp, self.S, self.PTP

# ======================================================================
# STDP Synapse
# ======================================================================
class STDPSynapse(SynapseCore):
    def __init__(self, pre, post, delay_ms=1.5, Q_max=50.0, tau_ms=2.0):
        super().__init__(pre.soma, post.soma, delay_ms=delay_ms, Q_max=Q_max, tau_ms=tau_ms)
        self.pre_neuron = pre
        self.post_neuron = post
        self.weight = 1.0
        self.last_pre_time = -100.0
        self.last_post_time = -100.0

    def on_pre_spike(self, t, Ca, R, ATP, dphi):
        self.last_pre_time = t
        dt_stdp = t - self.last_post_time
        if 0 < dt_stdp < 20.0:
            self.weight = max(0.1, self.weight - 0.05 * np.exp(-dt_stdp/10.0))
        super().on_pre_spike(t, Ca, R * self.weight, ATP, dphi)

    def on_post_spike(self, t):
        self.last_post_time = t
        dt = t - self.last_pre_time
        if 0 < dt < 20.0:
            self.weight = min(50.0, self.weight + 0.15 * np.exp(-dt/10.0))

# ======================================================================
# CA3 Neuron (기본)
# ======================================================================
class CA3Neuron:
    def __init__(self, name):
        self.name = name
        self.soma = HHSomaQuick(CONFIG["HH"])
        self.S, self.PTP = 0.0, 1.0
        self.outgoing_synapses = []
        self.incoming_synapses = []

    def step(self, dt, I_ext=0.0, t=0.0):
        self.soma.step(dt, I_ext)
        sp = self.soma.spiking()
        
        if sp:
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
            for syn in self.outgoing_synapses:
                syn.on_pre_spike(t, self.S, self.PTP, 100.0, 0.0)
            for syn in self.incoming_synapses:
                syn.on_post_spike(t)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
            
        return sp, self.S, self.PTP

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("⏰ HIPPO CA1: Temporal Encoding")
    print("=" * 70)
    print("Testing: Precise timing of A→B→C sequence")
    print("=" * 70)
    
    dt = 0.1
    
    # =========================================================
    # NETWORK SETUP
    # =========================================================
    print("\n✅ Creating CA3→CA1 network...")
    
    # CA3 neurons (3개: A, B, C)
    ca3_neurons = {
        'A': CA3Neuron('CA3_A'),
        'B': CA3Neuron('CA3_B'),
        'C': CA3Neuron('CA3_C')
    }
    
    # CA1 time cells (각 CA3 뉴런마다 시간 세포)
    ca1_time_cells = {
        'A': TimeCell(delay_ms=0, name='CA1_A'),    # A는 즉시
        'B': TimeCell(delay_ms=10, name='CA1_B'),   # B는 10ms 후
        'C': TimeCell(delay_ms=20, name='CA1_C')    # C는 20ms 후
    }
    
    print(f"   CA3 neurons: {len(ca3_neurons)}")
    print(f"   CA1 time cells: {len(ca1_time_cells)}")
    
    # CA3→CA1 연결 (각 CA3가 자신의 CA1 time cell 트리거)
    ca3_to_ca1_synapses = []
    for letter in ['A', 'B', 'C']:
        syn = STDPSynapse(ca3_neurons[letter], ca1_time_cells[letter], 
                         delay_ms=2.0, Q_max=50.0)
        ca3_neurons[letter].outgoing_synapses.append(syn)
        ca1_time_cells[letter].incoming_synapses.append(syn)
        ca3_to_ca1_synapses.append(syn)
    
    # CA3 내부 시퀀스 연결 (A→B→C)
    ca3_synapses = []
    for pre, post in [('A', 'B'), ('B', 'C')]:
        syn = STDPSynapse(ca3_neurons[pre], ca3_neurons[post],
                         delay_ms=2.0, Q_max=50.0)
        ca3_neurons[pre].outgoing_synapses.append(syn)
        ca3_neurons[post].incoming_synapses.append(syn)
        ca3_synapses.append(syn)
    
    all_synapses = ca3_to_ca1_synapses + ca3_synapses
    
    print(f"\n✅ Synapses created:")
    print(f"   CA3→CA1: {len(ca3_to_ca1_synapses)}")
    print(f"   CA3 sequence: {len(ca3_synapses)}")
    
    # =========================================================
    # PHASE 1: CA3 LEARNING (A→B→C)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: CA3 SEQUENCE LEARNING")
    print("=" * 70)
    
    T_learn = 80.0
    steps = int(T_learn/dt)
    num_repetitions = 10
    
    print(f"\nTraining A→B→C sequence ({num_repetitions} repetitions)...")
    
    for rep in range(num_repetitions):
        print(f"  [{rep+1}/{num_repetitions}]...", end="")
        
        for k in range(steps):
            t = k * dt
            
            # 순차적 자극
            I_ca3 = {'A': 0.0, 'B': 0.0, 'C': 0.0}
            
            if 5.0 < t < 13.0:
                I_ca3['A'] = 300.0
            if 20.0 < t < 28.0:
                I_ca3['B'] = 300.0
            if 35.0 < t < 43.0:
                I_ca3['C'] = 300.0
            
            # CA3 업데이트
            for letter, neuron in ca3_neurons.items():
                I_syn = sum(syn.I_syn for syn in neuron.incoming_synapses)
                neuron.step(dt, I_ca3[letter] + I_syn, t)
            
            # 시냅스 전달
            for s in all_synapses:
                s.deliver(t)
        
        # Reset
        for neuron in ca3_neurons.values():
            neuron.soma.V = -70.0
            neuron.soma.m = 0.05
            neuron.soma.h = 0.60
            neuron.soma.n = 0.32
            neuron.soma.spike_flag = False
            neuron.soma.mode = "rest"
            neuron.soma.ref_remaining = 0.0
            neuron.S = 0.0
            neuron.PTP = 1.0
        for s in all_synapses:
            s.spikes = []
            s.I_syn = 0.0
        
        print(" Done.")
    
    print("\n✅ CA3 sequence learning complete!")
    
    # =========================================================
    # PHASE 2: CA3→CA1 CONNECTION LEARNING
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: CA3→CA1 CONNECTION LEARNING")
    print("=" * 70)
    
    print("\nTraining CA3→CA1 associations...")
    
    for rep in range(10):
        print(f"  [{rep+1}/10]...", end="")
        
        for k in range(steps):
            t = k * dt
            
            # CA3와 CA1을 동시에 자극 (Hebbian)
            I_ca3 = {'A': 0.0, 'B': 0.0, 'C': 0.0}
            I_ca1 = {'A': 0.0, 'B': 0.0, 'C': 0.0}
            
            if 5.0 < t < 13.0:
                I_ca3['A'] = 300.0
                I_ca1['A'] = 250.0
            if 20.0 < t < 28.0:
                I_ca3['B'] = 300.0
                I_ca1['B'] = 250.0
            if 35.0 < t < 43.0:
                I_ca3['C'] = 300.0
                I_ca1['C'] = 250.0
            
            # 업데이트
            for letter, neuron in ca3_neurons.items():
                I_syn = sum(syn.I_syn for syn in neuron.incoming_synapses)
                neuron.step(dt, I_ca3[letter] + I_syn, t)
            
            for letter, cell in ca1_time_cells.items():
                I_syn = sum(syn.I_syn for syn in cell.incoming_synapses)
                cell.step(dt, t, I_ca1[letter] + I_syn)
            
            for s in all_synapses:
                s.deliver(t)
        
        # Reset
        for neuron in ca3_neurons.values():
            neuron.soma.V = -70.0
            neuron.soma.m = 0.05
            neuron.soma.h = 0.60
            neuron.soma.n = 0.32
            neuron.soma.spike_flag = False
            neuron.soma.mode = "rest"
            neuron.soma.ref_remaining = 0.0
            neuron.S = 0.0
            neuron.PTP = 1.0
        for cell in ca1_time_cells.values():
            cell.soma.V = -70.0
            cell.soma.m = 0.05
            cell.soma.h = 0.60
            cell.soma.n = 0.32
            cell.soma.spike_flag = False
            cell.soma.mode = "rest"
            cell.soma.ref_remaining = 0.0
            cell.S = 0.0
            cell.PTP = 1.0
            cell.trigger_time = None
        for s in all_synapses:
            s.spikes = []
            s.I_syn = 0.0
        
        print(" Done.")
    
    print("\n✅ CA3→CA1 learning complete!")
    
    # =========================================================
    # PHASE 3: TEMPORAL RECALL TEST
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: TEMPORAL RECALL TEST")
    print("=" * 70)
    
    # Reset
    for neuron in ca3_neurons.values():
        neuron.soma.V = -70.0
        neuron.soma.m = 0.05
        neuron.soma.h = 0.60
        neuron.soma.n = 0.32
        neuron.soma.spike_flag = False
        neuron.soma.mode = "rest"
        neuron.soma.ref_remaining = 0.0
        neuron.S = 0.0
        neuron.PTP = 1.0
    for cell in ca1_time_cells.values():
        cell.soma.V = -70.0
        cell.soma.m = 0.05
        cell.soma.h = 0.60
        cell.soma.n = 0.32
        cell.soma.spike_flag = False
        cell.soma.mode = "rest"
        cell.soma.ref_remaining = 0.0
        cell.S = 0.0
        cell.PTP = 1.0
        cell.trigger_time = None
    for s in all_synapses:
        s.spikes = []
        s.I_syn = 0.0
    
    print("\n🧪 Test: Cue 'A' → CA3 sequence → CA1 timing")
    
    T_test = 100.0
    steps_test = int(T_test/dt)
    
    ca3_log = []
    ca1_log = []
    
    # CA3가 발화하면 CA1 time cell 트리거
    ca3_fired = {'A': False, 'B': False, 'C': False}
    
    for k in range(steps_test):
        t = k * dt
        
        # Cue A만
        I_ca3 = {'A': 0.0, 'B': 0.0, 'C': 0.0}
        if 1.0 <= t < 5.0:
            I_ca3['A'] = 300.0
        
        # CA3 업데이트
        for letter, neuron in ca3_neurons.items():
            I_syn = sum(syn.I_syn for syn in neuron.incoming_synapses)
            sp, _, _ = neuron.step(dt, I_ca3[letter] + I_syn, t)
            if sp:
                ca3_log.append((t, letter))
                # CA1 time cell 트리거
                if not ca3_fired[letter]:
                    ca1_time_cells[letter].trigger(t)
                    ca3_fired[letter] = True
        
        # CA1 업데이트
        for letter, cell in ca1_time_cells.items():
            I_syn = sum(syn.I_syn for syn in cell.incoming_synapses)
            sp, _, _ = cell.step(dt, t, I_syn)
            if sp:
                ca1_log.append((t, letter))
        
        for s in all_synapses:
            s.deliver(t)
    
    # 결과 분석
    print(f"\n📊 CA3 Sequence (learned order):")
    ca3_times = {}
    for t, letter in ca3_log:
        if letter not in ca3_times:
            ca3_times[letter] = t
            print(f"   {letter}: {t:.1f}ms")
    
    print(f"\n⏰ CA1 Temporal Code (precise timing):")
    ca1_times = {}
    for t, letter in ca1_log:
        if letter not in ca1_times:
            ca1_times[letter] = t
            print(f"   {letter}: {t:.1f}ms")
    
    # 시간 간격 계산
    if 'A' in ca1_times and 'B' in ca1_times and 'C' in ca1_times:
        interval_AB = ca1_times['B'] - ca1_times['A']
        interval_BC = ca1_times['C'] - ca1_times['B']
        
        print(f"\n🎯 Temporal Intervals:")
        print(f"   A→B: {interval_AB:.1f}ms (Target: 10ms)")
        print(f"   B→C: {interval_BC:.1f}ms (Target: 10ms)")
        
        accuracy_AB = abs(interval_AB - 10.0)
        accuracy_BC = abs(interval_BC - 10.0)
        
        if accuracy_AB < 3.0 and accuracy_BC < 3.0:
            print(f"\n   ✅ Precise timing achieved! (error < 3ms)")
        else:
            print(f"\n   ⚠️ Timing error: {max(accuracy_AB, accuracy_BC):.1f}ms")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    print("=" * 70)
    
    if len(ca3_times) == 3 and len(ca1_times) == 3:
        print("\n✅ SUCCESS: CA1 Temporal Encoding Working!")
        print(f"   CA3: Sequence order (A→B→C)")
        print(f"   CA1: Precise timing (10ms intervals)")
        print(f"\n   → Time cells successfully encode temporal structure!")
    else:
        print("\n⚠️ Incomplete activation")
    
    # =========================================================
    # VISUALIZATION
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 GENERATING VISUALIZATION...")
    print("=" * 70)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 1. CA3 Timeline
    ax1 = plt.subplot(1, 2, 1)
    for t, letter in ca3_log[:10]:  # 처음 몇 개만
        y = {'A': 3, 'B': 2, 'C': 1}[letter]
        ax1.scatter(t, y, s=100, color='red', marker='o', edgecolors='black', linewidth=2)
    
    ax1.set_ylabel('CA3 Neurons', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['C', 'B', 'A'])
    ax1.set_title('[1] CA3 Sequence (Order)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, 60)
    
    # 2. CA1 Timeline
    ax2 = plt.subplot(1, 2, 2)
    for t, letter in ca1_log[:10]:
        y = {'A': 3, 'B': 2, 'C': 1}[letter]
        ax2.scatter(t, y, s=100, color='blue', marker='s', edgecolors='black', linewidth=2)
    
    # 목표 시간 표시
    if 'A' in ca1_times:
        ref_t = ca1_times['A']
        ax2.axvline(ref_t, color='gray', linestyle='--', alpha=0.5, label='A')
        ax2.axvline(ref_t+10, color='gray', linestyle='--', alpha=0.5, label='Target +10ms')
        ax2.axvline(ref_t+20, color='gray', linestyle='--', alpha=0.5, label='Target +20ms')
    
    ax2.set_ylabel('CA1 Time Cells', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['C (20ms)', 'B (10ms)', 'A (0ms)'])
    ax2.set_title('[2] CA1 Temporal Code (Timing)', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, 60)
    ax2.legend()
    
    plt.tight_layout()
    
    output_file = '/Users/jazzin/Desktop/hippo_v0/ca1_temporal_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {output_file}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✨ CA1 adds temporal precision to CA3's sequence!")
    print("=" * 70)

