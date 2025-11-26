import numpy as np
import random

# ✅ 핵심 엔진 임포트
from v3_event import CONFIG, HHSomaQuick, SynapseCore

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
        self.pre_neuron = pre # Pre 뉴런 참조 필요
        self.post_neuron = post # Post 뉴런 참조 필요
        self.weight = 1.0     # 가중치 (기본 1.0)
        self.last_pre_time = -100.0
        self.last_post_time = -100.0

    def on_pre_spike(self, t, Ca, R, ATP, dphi):
        self.last_pre_time = t
        # STDP Update: Post가 최근에 발화했는지 확인 (Post -> Pre: LTD)
        dt_stdp = t - self.last_post_time
        if 0 < dt_stdp < 20.0: # Post가 먼저 튄 경우 (역방향)
            self.weight = max(0.1, self.weight - 0.3 * np.exp(-dt_stdp/10.0))  # 0.5 → 0.3 (약화 감소)
            
        # 신호 전달 (가중치 적용)
        super().on_pre_spike(t, Ca, R * self.weight, ATP, dphi)

    def on_post_spike(self, t):
        self.last_post_time = t
        # STDP Update: Pre가 최근에 발화했는지 확인 (Pre -> Post: LTP)
        dt = t - self.last_pre_time
        if 0 < dt < 20.0: # Pre가 먼저 튄 경우 (순방향)
            # 학습률 증가: 1.0 → 1.5, 최대 가중치 유지: 5.0
            self.weight = min(5.0, self.weight + 1.5 * np.exp(-dt/10.0))  # 1.0 → 1.5 (강화 증가)

# ======================================================================
# 2. Neuron with Post-Spike Hook
# ======================================================================
class SequenceNeuron:
    def __init__(self, name):
        self.name = name
        self.soma = HHSomaQuick(CONFIG["HH"])
        self.S, self.PTP = 0.0, 1.0
        self.outgoing_synapses = [] # 내가 Pre인 시냅스들
        self.incoming_synapses = [] # 내가 Post인 시냅스들

    def step(self, dt, I_ext=0.0, t=0.0):
        self.soma.step(dt, I_ext)
        sp = self.soma.spiking()
        
        if sp:
            # PTP 업데이트
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
            
            # 1. Outgoing Synapse에 신호 전달 (Pre Spike)
            for syn in self.outgoing_synapses:
                syn.on_pre_spike(t, self.S, self.PTP, 100.0, 0.0)

            # 2. Incoming Synapse에 알림 (Post Spike -> STDP)
            for syn in self.incoming_synapses:
                syn.on_post_spike(t)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
            
        return sp, self.S, self.PTP

# ======================================================================
# 3. Sequence Learning Simulation
# ======================================================================
def run_sequence_memory(N=20, dt=0.1):
    random.seed(42); np.random.seed(42)
    print(f"\n🧠 HIPPOCAMPUS SEQUENCE MEMORY (A -> B -> C)")
    print("=" * 70)

    neurons = [SequenceNeuron(f"N{i}") for i in range(N)]
    
    # --- 패턴 정의 (먼저 선언) ---
    seq_A = [0, 1]
    seq_B = [5, 6]
    seq_C = [10, 11]
    
    # ✅ 선택적 연결 전략: A→B, B→C 경로만 생성 (간섭 최소화)
    synapses = []
    
    # A→B 연결 (4개: 0→5, 0→6, 1→5, 1→6)
    for i in seq_A:
        for j in seq_B:
            syn = STDPSynapse(neurons[i], neurons[j], delay_ms=2.0, Q_max=20.0)
            neurons[i].outgoing_synapses.append(syn)
            neurons[j].incoming_synapses.append(syn)
            synapses.append(syn)
    
    # B→C 연결 (4개: 5→10, 5→11, 6→10, 6→11)
    for i in seq_B:
        for j in seq_C:
            syn = STDPSynapse(neurons[i], neurons[j], delay_ms=2.0, Q_max=20.0)
            neurons[i].outgoing_synapses.append(syn)
            neurons[j].incoming_synapses.append(syn)
            synapses.append(syn)

    print(f"Network Ready: {len(synapses)} Selective STDP Synapses (A→B→C pathway).")

    # =========================================================
    # PHASE 1: SEQUENCE LEARNING (반복 학습)
    # =========================================================
    print("\n=== PHASE 1: LEARNING (Time-Lagged Input, 15 repetitions) ===")
    
    num_repeats = 15  # 10 → 15 증가 (충분한 학습)
    for rep in range(num_repeats):
        print(f"  Repetition {rep+1}/{num_repeats}...", end="")
        T_learn = 80.0  # 50 → 80 증가 (더 긴 간격)
        steps = int(T_learn/dt)

        for k in range(steps):
            t = k * dt
            
            # 시간차 자극: A(5ms) -> B(20ms) -> C(24ms, synapse-assisted)
            I = np.zeros(N)
            if 5.0 < t < 8.0: 
                for i in seq_A: I[i] = 250.0
            if 20.0 < t < 23.0: 
                for i in seq_B: I[i] = 250.0
            if 24.0 < t < 27.0:  # ✅ B 직후 (23ms + 1ms) - STDP LTP 유도
                for i in seq_C: I[i] = 120.0  # ✅ 시냅스 보조 수준 (B→C 신호 + 약한 자극)
            
            # 뉴런 업데이트 & STDP
            for i in range(N):
                sp, _, _ = neurons[i].step(dt, I[i], t)
            
            # 시냅스 전달
            for s in synapses: s.deliver(t)
        
        # 반복 간 휴식
        for _ in range(100):
            for i in range(N):
                neurons[i].step(dt, 0.0, t)
            for s in synapses: s.deliver(t)
        
        print(" Done.")

    print("\n✅ Sequence Learning Complete.")
    
    # 학습된 가중치 확인 (A→B, B→C 연결)
    print("\n🔍 STDP Weights Check:")
    for i in seq_A:
        for j in seq_B:
            for syn in neurons[i].outgoing_synapses:
                if syn.post_neuron == neurons[j]:
                    print(f"  N{i}→N{j}: weight={syn.weight:.2f}")
    for i in seq_B:
        for j in seq_C:
            for syn in neurons[i].outgoing_synapses:
                if syn.post_neuron == neurons[j]:
                    print(f"  N{i}→N{j}: weight={syn.weight:.2f}")

    # =========================================================
    # PHASE 2: RESET
    # =========================================================
    print("\n=== PHASE 2: RESET ===")
    for n in neurons: 
        n.soma.V=-70; n.soma.spike_flag=False; n.soma.mode="rest"
        n.S = 0.0; n.PTP = 1.0  # ✅ S, PTP도 초기화
    for s in synapses: s.spikes=[]; s.I_syn=0
    print("Reset Done (including S/PTP).")

    # =========================================================
    # PHASE 3: RECALL (Sequence Completion)
    # =========================================================
    print("\n=== PHASE 3: RECALL (Cue: A only) ===")
    print(f"Cue: {seq_A} -> Expecting: {seq_B} -> {seq_C}")
    
    T_test = 60.0
    steps = int(T_test/dt)
    logs = []
    syn_currents = []  # ✅ 시냅스 전류 기록

    for k in range(steps):
        t = k * dt
        
        # Cue A only (매우 짧고 강하게 - 단일 펄스)
        I = np.zeros(N)
        if 1.0 < t < 2.0:  # 3.0 → 1ms 펄스 (A 발화 후 즉시 종료)
            for i in seq_A: I[i] = 300.0  # 250 → 300 (더 강하게)
            
        spikes = []
        for i in range(N):
            # ✅ 시냅스 전류 합산
            I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
            sp, _, PTP = neurons[i].step(dt, I[i] + I_syn_total, t)
            if sp: 
                spikes.append(i)
        
        for s in synapses: s.deliver(t)
        
        # ✅ B 뉴런의 시냅스 전류 기록
        if t > 1.0 and t < 10.0:
            for syn in synapses:
                if syn.post_neuron in [neurons[i] for i in seq_B]:
                    if syn.I_syn > 0:
                        syn_currents.append((t, syn.I_syn))
        
        if spikes: logs.append((t, spikes))

    # --- 결과 시각화 ---
    print("\n[Sequence Replay Log]")
    print("Time | Active Neurons")
    print("-" * 40)
    
    # 패턴별로 분류
    A_times, B_times, C_times = [], [], []
    for t, ids in logs:
        if t > 3.0:  # Cue 이후
            if any(x in seq_A for x in ids): A_times.append(t)
            if any(x in seq_B for x in ids): B_times.append(t)
            if any(x in seq_C for x in ids): C_times.append(t)
    
    # 요약 출력
    print(f"✅ Pattern A: {len(A_times)} spikes (First: {A_times[0] if A_times else 'None'}ms)")
    print(f"{'✅' if B_times else '❌'} Pattern B: {len(B_times)} spikes (First: {B_times[0] if B_times else 'None'}ms)")
    print(f"{'✅' if C_times else '❌'} Pattern C: {len(C_times)} spikes (First: {C_times[0] if C_times else 'None'}ms)")
    
    # ✅ 시냅스 전류 확인
    print(f"\n🔍 Synaptic Currents to B: {len(syn_currents)} events")
    if syn_currents:
        print(f"   First current: {syn_currents[0][0]:.1f}ms, I={syn_currents[0][1]:.1f}pA")
        print(f"   Max current: {max(c[1] for c in syn_currents):.1f}pA")
    else:
        print("   ⚠️ NO synaptic input to B detected!")
    
    # 상세 로그 (처음 20개)
    print("\nDetailed Log (First 20 events after cue):")
    count = 0
    for t, ids in logs:
        if t > 3.0 and count < 20:
            ids_str = str(ids)
            if any(x in seq_B for x in ids): ids_str += " ✨ Pattern B!"
            if any(x in seq_C for x in ids): ids_str += " ✨ Pattern C!"
            print(f"{t:4.1f}ms | {ids_str}")
            count += 1

if __name__ == "__main__":
    run_sequence_memory()