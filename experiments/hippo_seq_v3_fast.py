import numpy as np
import random

# ✅ 핵심 엔진 임포트
from v4_event import CONFIG, HHSomaQuick, SynapseCore

# ======================================================================
# 1. STDP Synapse (시간차 학습 기능 추가)
# ======================================================================
class STDPSynapse(SynapseCore):
    """
    Spike-Timing-Dependent Plasticity (STDP) Synapse
    - Pre가 Post보다 먼저 발화하면 강화 (LTP)
    - Post가 Pre보다 먼저 발화하면 약화 (LTD)
    """
    def __init__(self, pre, post, delay_ms=1.5, Q_max=10.0, tau_ms=2.0):
        super().__init__(pre.soma, post.soma, delay_ms=delay_ms, Q_max=Q_max, tau_ms=tau_ms)
        self.pre_neuron = pre
        self.post_neuron = post
        self.weight = 1.0
        self.last_pre_time = -100.0
        self.last_post_time = -100.0

    def on_pre_spike(self, t, Ca, R, ATP, dphi):
        self.last_pre_time = t
        # STDP Update: LTD (Post가 먼저)
        dt_stdp = t - self.last_post_time
        if 0 < dt_stdp < 20.0:
            self.weight = max(0.1, self.weight - 0.3 * np.exp(-dt_stdp/10.0))
        
        # 신호 전달 (가중치 적용)
        super().on_pre_spike(t, Ca, R * self.weight, ATP, dphi)

    def on_post_spike(self, t):
        self.last_post_time = t
        # STDP Update: LTP (Pre가 먼저)
        dt = t - self.last_pre_time
        if 0 < dt < 20.0:
            self.weight = min(5.0, self.weight + 1.5 * np.exp(-dt/10.0))

# ======================================================================
# 2. Neuron with Post-Spike Hook
# ======================================================================
class SequenceNeuron:
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
# 3. Long Sequence Learning (A→B→C→D→E→F→G→H)
# ======================================================================
def run_long_sequence_memory(N=20, dt=0.1):
    random.seed(42); np.random.seed(42)
    print(f"\n🧠 HIPPOCAMPUS LONG SEQUENCE MEMORY (v3)")
    print("=" * 70)
    print("Testing: Long sequence A→B→C→D→E→F→G→H")
    print("=" * 70)

    neurons = [SequenceNeuron(f"N{i}") for i in range(N)]
    
    # --- 긴 시퀀스 정의 (A부터 H까지 8단계, 2-뉴런 패턴) ---
    # ✅ 각 단계를 2개 뉴런으로 표현하여 더 강한 학습 유도
    sequence = {
        "A": [0, 1],
        "B": [2, 3],
        "C": [4, 5],
        "D": [6, 7],
        "E": [8, 9],
        "F": [10, 11],
        "G": [12, 13],
        "H": [14, 15]
    }
    
    seq_order = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    # ✅ 연쇄 연결 생성 (A→B→C→D→E→F→G→H)
    # 각 연결마다 Q_max를 점진적으로 증가시켜 장거리 전파 보장
    synapses = []
    
    # Q_max 증가 전략: 모두 50.0으로 통일 (v4_event 최적화)
    Q_values = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    
    for i in range(len(seq_order) - 1):
        pre_name = seq_order[i]
        post_name = seq_order[i + 1]
        
        pre_neurons = sequence[pre_name]
        post_neurons = sequence[post_name]
        
        Q_max_val = Q_values[i] if i < len(Q_values) else 50.0
        
        for pre_idx in pre_neurons:
            for post_idx in post_neurons:
                syn = STDPSynapse(neurons[pre_idx], neurons[post_idx], delay_ms=2.0, Q_max=Q_max_val)
                neurons[pre_idx].outgoing_synapses.append(syn)
                neurons[post_idx].incoming_synapses.append(syn)
                synapses.append(syn)
    
    print(f"\n✅ Network Ready:")
    print(f"   Total Synapses: {len(synapses)}")
    path_str = " → ".join([f"{name}{sequence[name]}" for name in seq_order])
    print(f"   Path: {path_str}")

    # =========================================================
    # PHASE 1: STEP-BY-STEP LEARNING (단계별 독립 학습)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: STEP-BY-STEP LEARNING")
    print("=" * 70)
    print("  Strategy: Learn each connection independently")
    print("  (like v2's interleaved training)")
    print()
    
    # ✅ v2처럼 각 연결을 독립적으로 학습
    # A→B, B→C, C→D, ... 를 따로따로 반복 학습
    
    num_repeats = 3  # 10→3 (빠른 실험)
    T_pair = 50.0  # 80→50 (각 쌍당 학습 시간 단축)
    steps_pair = int(T_pair/dt)
    
    # 연결 쌍 정의
    pairs = [
        ("A", "B", (5.0, 13.0, 250.0), (20.0, 28.0, 300.0)),
        ("B", "C", (5.0, 13.0, 250.0), (20.0, 28.0, 300.0)),
        ("C", "D", (5.0, 13.0, 250.0), (20.0, 28.0, 300.0)),
        ("D", "E", (5.0, 13.0, 250.0), (20.0, 28.0, 300.0)),
        ("E", "F", (5.0, 13.0, 250.0), (20.0, 28.0, 300.0)),
        ("F", "G", (5.0, 13.0, 250.0), (20.0, 28.0, 300.0)),
        ("G", "H", (5.0, 13.0, 250.0), (20.0, 28.0, 300.0))
    ]
    
    for rep in range(num_repeats):
        print(f"  Cycle {rep+1}/{num_repeats}:")
        
        for pre_name, post_name, pre_stim, post_stim in pairs:
            print(f"    Training {pre_name}→{post_name}...", end="")
            
            pre_neurons = sequence[pre_name]
            post_neurons = sequence[post_name]
            
            for k in range(steps_pair):
                t = k * dt
                
                I = np.zeros(N)
                # Pre 자극
                if pre_stim[0] < t < pre_stim[1]:
                    for i in pre_neurons:
                        I[i] = pre_stim[2]
                # Post 자극 (더 늦게, 더 강하게)
                if post_stim[0] < t < post_stim[1]:
                    for i in post_neurons:
                        I[i] = post_stim[2]
                
                # 뉴런 업데이트
                for i in range(N):
                    I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
                    neurons[i].step(dt, I[i] + I_syn_total, t)
                
                # 시냅스 전달
                for s in synapses:
                    s.deliver(t)
            
            # 쌍 간 세척 (충분히 길게)
            for _ in range(200):
                for i in range(N):
                    neurons[i].step(dt, 0.0, t)
                for s in synapses:
                    s.deliver(t)
            
            # Reset (각 쌍 학습 후 - 완전 초기화)
            for n in neurons:
                n.soma.V = -70.0
                n.soma.m = 0.05
                n.soma.h = 0.60
                n.soma.n = 0.32
                n.soma.spike_flag = False
                n.soma.mode = "rest"
                n.soma.ref_remaining = 0.0
                n.S = 0.0
                n.PTP = 1.0
            # ✅ 시냅스도 완전 초기화!
            for s in synapses:
                s.spikes = []
                s.I_syn = 0.0
                if hasattr(s, 'Ca'):
                    s.Ca = 0.0
                if hasattr(s, 'R'):
                    s.R = 1.0
            
            print(" Done.")
    
    print("\n✅ Long Sequence Learning Complete.")
    
    # 학습된 가중치 확인
    print("\n🔍 STDP Weights Check:")
    for i in range(len(seq_order) - 1):
        pre_name = seq_order[i]
        post_name = seq_order[i + 1]
        
        pre_neurons_list = sequence[pre_name]
        post_neurons_list = sequence[post_name]
        
        for pre_idx in pre_neurons_list:
            for post_idx in post_neurons_list:
                for syn in neurons[pre_idx].outgoing_synapses:
                    if syn.post_neuron == neurons[post_idx]:
                        print(f"  {pre_name}(N{pre_idx})→{post_name}(N{post_idx}): weight={syn.weight:.2f}")

    # =========================================================
    # PHASE 2: RESET
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: RESET")
    print("=" * 70)
    for n in neurons:
        n.soma.V = -70.0
        
        # ✅ HH 게이트 변수 초기화 (핵심!)
        n.soma.m = 0.05   # 나트륨 활성화 게이트
        n.soma.h = 0.60   # 나트륨 불활성화 게이트 (매우 중요!)
        n.soma.n = 0.32   # 칼륨 활성화 게이트
        
        n.soma.spike_flag = False
        n.soma.mode = "rest"
        n.soma.ref_remaining = 0.0
        n.S = 0.0
        n.PTP = 1.0
    for s in synapses:
        s.spikes = []
        s.I_syn = 0
    print("✅ Reset Done (including S/PTP/ref/gates).")

    # =========================================================
    # PHASE 3: SEQUENTIAL RECALL
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: SEQUENTIAL RECALL (Cue: A only)")
    print("=" * 70)
    
    cue = sequence["A"]
    print(f"\n🧪 Test: Cue A{cue} → Expecting full sequence B→C→D→E→F→G→H")
    
    T_test = 100.0  # 더 긴 테스트 시간
    steps = int(T_test/dt)
    
    # Recall
    logs = []
    for k in range(steps):
        t = k * dt
        
        # Cue A only
        I = np.zeros(N)
        # ✅ 수정: 1.0 <= t < 10.0 (경계값 포함!)
        if 1.0 <= t < 10.0:
            for i in cue:
                I[i] = 300.0
        
        spikes = []
        for i in range(N):
            I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
            sp, _, _ = neurons[i].step(dt, I[i] + I_syn_total, t)
            if sp:
                spikes.append(i)
        
        for s in synapses:
            s.deliver(t)
        
        if spikes:
            logs.append((t, spikes))
    
    # 분석: 각 단계의 활성화 확인
    print("\n📊 Activation Analysis:")
    activation_counts = {name: 0 for name in seq_order}
    first_activation = {name: None for name in seq_order}
    
    for t, ids in logs:
        for name in seq_order:
            if any(x in sequence[name] for x in ids):
                activation_counts[name] += 1
                if first_activation[name] is None:
                    first_activation[name] = t
    
    # 결과 출력
    for name in seq_order:
        count = activation_counts[name]
        first_t = first_activation[name]
        status = "✅" if count > 0 else "❌"
        first_str = f"{first_t:.1f}ms" if first_t else "None"
        print(f"   {status} Pattern {name}: {count} spikes (First: {first_str})")
    
    # 시퀀스 로그 (타임라인)
    print("\n🎬 Sequence Timeline (First 30 events):")
    print("Time | Active Patterns")
    print("-" * 40)
    
    count = 0
    for t, ids in logs:
        if t > 3.0 and count < 30:
            active_patterns = []
            for name in seq_order:
                if any(x in sequence[name] for x in ids):
                    active_patterns.append(name)
            
            if active_patterns:
                pattern_str = ", ".join(active_patterns)
                print(f"{t:4.1f}ms | {pattern_str}")
                count += 1
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for count in activation_counts.values() if count > 0)
    total_stages = len(seq_order)
    
    print(f"\n🎯 Activated Stages: {success_count}/{total_stages}")
    
    # 시퀀스 순서 확인
    sequence_order_correct = True
    prev_time = 0
    for i, name in enumerate(seq_order[1:], 1):  # B부터 시작
        curr_time = first_activation[name]
        if curr_time is None:
            sequence_order_correct = False
            break
        if curr_time < prev_time:
            sequence_order_correct = False
            break
        prev_time = curr_time
    
    if success_count == total_stages and sequence_order_correct:
        print("\n🎉 Perfect! Complete long sequence recall!")
        print("   ✅ All 8 stages activated")
        print("   ✅ Correct sequential order")
        print(f"   ✅ Total propagation: {first_activation[seq_order[-1]] - first_activation[seq_order[1]]:.1f}ms")
    else:
        print(f"\n⚠️ Partial success: {success_count}/{total_stages} stages activated")
        if not sequence_order_correct:
            print("   ❌ Sequence order violated")

if __name__ == "__main__":
    run_long_sequence_memory()

