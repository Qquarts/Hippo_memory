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
    def __init__(self, pre, post, delay_ms=1.5, Q_max=30.0, tau_ms=2.0):
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
            self.weight = max(0.1, self.weight - 0.1 * np.exp(-dt_stdp/10.0))  # 0.3→0.1 (약화 감소)
        
        # 신호 전달 (가중치 적용)
        super().on_pre_spike(t, Ca, R * self.weight, ATP, dphi)

    def on_post_spike(self, t):
        self.last_post_time = t
        # STDP Update: LTP (Pre가 먼저)
        dt = t - self.last_pre_time
        if 0 < dt < 20.0:
            self.weight = min(10.0, self.weight + 2.0 * np.exp(-dt/10.0))  # 5.0→10.0 상한, 1.5→2.0 학습률

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
# 3. Multi-Sequence Learning Simulation
# ======================================================================
def run_multi_sequence_memory(N=20, dt=0.1):
    random.seed(42); np.random.seed(42)
    print(f"\n🧠 HIPPOCAMPUS MULTI-SEQUENCE MEMORY (v2)")
    print("=" * 70)
    print("Testing: Multiple sequences in one network (no interference)")
    print("=" * 70)

    neurons = [SequenceNeuron(f"N{i}") for i in range(N)]
    
    # --- 다중 시퀀스 정의 (4개 독립 경로) ---
    sequences = {
        "Seq1": {
            "A": [0, 1],
            "B": [5, 6],
            "C": [10, 11]
        },
        "Seq2": {
            "A": [2, 3],
            "B": [7, 8],
            "C": [12, 13]
        },
        "Seq3": {
            "A": [4],      # 단일 뉴런
            "B": [9],
            "C": [14]
        },
        "Seq4": {
            "A": [15],     # 단일 뉴런
            "B": [16],
            "C": [17]
        }
    }
    
    # ✅ 독립 경로 생성 (간섭 방지)
    synapses_by_seq = {}
    total_synapses = []
    
    for seq_name, seq_data in sequences.items():
        seq_A, seq_B, seq_C = seq_data["A"], seq_data["B"], seq_data["C"]
        seq_synapses = []
        
        # A→B 연결
        for i in seq_A:
            for j in seq_B:
                syn = STDPSynapse(neurons[i], neurons[j], delay_ms=2.0, Q_max=50.0)  # ✅ 20→50 (파워 업!)
                neurons[i].outgoing_synapses.append(syn)
                neurons[j].incoming_synapses.append(syn)
                seq_synapses.append(syn)
                total_synapses.append(syn)
        
        # B→C 연결
        for i in seq_B:
            for j in seq_C:
                syn = STDPSynapse(neurons[i], neurons[j], delay_ms=2.0, Q_max=50.0)  # ✅ 30→50 (확실하게!)
                neurons[i].outgoing_synapses.append(syn)
                neurons[j].incoming_synapses.append(syn)
                seq_synapses.append(syn)
                total_synapses.append(syn)
        
        synapses_by_seq[seq_name] = seq_synapses
    
    print(f"\n✅ Network Ready:")
    print(f"   Total Synapses: {len(total_synapses)}")
    for seq_name, seq_data in sequences.items():
        print(f"   {seq_name}: {seq_data['A']} → {seq_data['B']} → {seq_data['C']} ({len(synapses_by_seq[seq_name])} synapses)")

    # =========================================================
    # PHASE 1: MULTI-SEQUENCE LEARNING (교대 학습)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LEARNING (Interleaved Training)")
    print("=" * 70)
    
    num_repeats = 10
    T_learn = 80.0
    steps = int(T_learn/dt)
    
    for rep in range(num_repeats):
        print(f"\n  Cycle {rep+1}/{num_repeats}:")
        
        # 각 시퀀스를 순차적으로 학습
        for seq_name, seq_data in sequences.items():
            seq_A, seq_B, seq_C = seq_data["A"], seq_data["B"], seq_data["C"]
            print(f"    Training {seq_name}...", end="")
            
            for k in range(steps):
                t = k * dt
                
                # 시간차 자극: A(5ms) -> B(20-28ms, 강화) -> C(32-40ms, 길게)
                # ✅ B를 충분히 자극하여 B→C 시냅스 학습 강화
                I = np.zeros(N)
                if 5.0 < t < 8.0:
                    for i in seq_A: I[i] = 250.0
                if 20.0 < t < 28.0:  # 23→28 (8ms 동안 강하게)
                    for i in seq_B: I[i] = 300.0  # 250→300 강화
                if 32.0 < t < 40.0:  # 더 길고 명확한 C 자극
                    for i in seq_C: I[i] = 200.0
                
                # 뉴런 업데이트 (전체 - 시냅스 전류 합산)
                for i in range(N):
                    I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
                    neurons[i].step(dt, I[i] + I_syn_total, t)
                
                # 시냅스 전달
                for s in total_synapses:
                    s.deliver(t)
            
            # 시퀀스 간 완전 세척 (간섭 제거)
            for _ in range(200):
                for i in range(N):
                    neurons[i].step(dt, 0.0, t)
                for s in total_synapses:
                    s.deliver(t)
            
            # Reset
            for n in neurons:
                n.soma.V = -70
                n.soma.spike_flag = False
                n.soma.mode = "rest"
                n.S = 0.0
                n.PTP = 1.0
            for s in total_synapses:
                s.spikes = []
                s.I_syn = 0
            
            print(" Done.")
    
    print("\n✅ Multi-Sequence Learning Complete.")
    
    # 학습된 가중치 확인
    print("\n🔍 STDP Weights Check:")
    for seq_name, seq_data in sequences.items():
        seq_A, seq_B, seq_C = seq_data["A"], seq_data["B"], seq_data["C"]
        print(f"\n  {seq_name}:")
        
        # A→B 가중치
        for i in seq_A:
            for j in seq_B:
                for syn in neurons[i].outgoing_synapses:
                    if syn.post_neuron == neurons[j]:
                        print(f"    N{i}→N{j}: weight={syn.weight:.2f}")
        
        # B→C 가중치
        for i in seq_B:
            for j in seq_C:
                for syn in neurons[i].outgoing_synapses:
                    if syn.post_neuron == neurons[j]:
                        print(f"    N{i}→N{j}: weight={syn.weight:.2f}")

    # =========================================================
    # PHASE 2: FINAL RESET
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: RESET")
    print("=" * 70)
    for n in neurons:
        n.soma.V = -70
        n.soma.spike_flag = False
        n.soma.mode = "rest"
        n.S = 0.0
        n.PTP = 1.0
    for s in total_synapses:
        s.spikes = []
        s.I_syn = 0
    print("✅ Reset Done (including S/PTP).")

    # =========================================================
    # PHASE 3: SELECTIVE RECALL (간섭 테스트)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: SELECTIVE RECALL (Interference Test)")
    print("=" * 70)
    
    T_test = 60.0
    steps = int(T_test/dt)
    
    results = {}
    
    for seq_name, seq_data in sequences.items():
        seq_A, seq_B, seq_C = seq_data["A"], seq_data["B"], seq_data["C"]
        cue = [seq_A[0]]
        
        print(f"\n🧪 Test {seq_name}: Cue {cue} → Expecting {seq_B}, {seq_C}")
        
        # Reset
        for n in neurons:
            n.soma.V = -70
            n.soma.spike_flag = False
            n.soma.mode = "rest"
            n.S = 0.0
            n.PTP = 1.0
        for s in total_synapses:
            s.spikes = []
            s.I_syn = 0
        
        # Recall
        logs = []
        for k in range(steps):
            t = k * dt
            
            # Cue
            I = np.zeros(N)
            if 1.0 < t < 2.0:
                for i in cue:
                    I[i] = 300.0
            
            spikes = []
            for i in range(N):
                I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
                sp, _, _ = neurons[i].step(dt, I[i] + I_syn_total, t)
                if sp:
                    spikes.append(i)
            
            for s in total_synapses:
                s.deliver(t)
            
            if spikes:
                logs.append((t, spikes))
        
        # 분석: 각 패턴의 활성화 확인
        A_active, B_active, C_active = 0, 0, 0
        for t, ids in logs:
            if t > 3.0:  # Cue 이후
                if any(x in seq_A for x in ids): A_active += 1
                if any(x in seq_B for x in ids): B_active += 1
                if any(x in seq_C for x in ids): C_active += 1
        
        # 간섭 체크: 다른 시퀀스가 활성화되었는지
        interference = {}
        for other_seq_name, other_seq_data in sequences.items():
            if other_seq_name == seq_name:
                continue
            
            other_A, other_B, other_C = other_seq_data["A"], other_seq_data["B"], other_seq_data["C"]
            other_active = 0
            for t, ids in logs:
                if t > 3.0:
                    if any(x in other_B for x in ids) or any(x in other_C for x in ids):
                        other_active += 1
            interference[other_seq_name] = other_active
        
        # 결과 저장
        results[seq_name] = {
            "A": A_active,
            "B": B_active,
            "C": C_active,
            "interference": interference
        }
        
        # 출력
        print(f"   📤 Pattern A: {A_active} spikes")
        print(f"   📤 Pattern B: {B_active} spikes {'✅' if B_active > 0 else '❌'}")
        print(f"   📤 Pattern C: {C_active} spikes {'✅' if C_active > 0 else '❌'}")
        
        for other_name, other_count in interference.items():
            status = "✅ No interference" if other_count == 0 else f"⚠️ {other_count} spikes"
            print(f"   🔍 {other_name} interference: {status}")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    print("=" * 70)
    
    success_count = 0
    for seq_name, result in results.items():
        B_ok = result["B"] > 0
        C_ok = result["C"] > 0
        no_interference = all(count == 0 for count in result["interference"].values())
        
        if B_ok and C_ok and no_interference:
            print(f"✅ {seq_name}: PERFECT (B✅ C✅ No interference✅)")
            success_count += 1
        else:
            issues = []
            if not B_ok: issues.append("B failed")
            if not C_ok: issues.append("C failed")
            if not no_interference: issues.append("Interference detected")
            print(f"❌ {seq_name}: FAILED ({', '.join(issues)})")
    
    print(f"\n🎯 Score: {success_count}/{len(sequences)}")
    
    if success_count == len(sequences):
        print("\n🎉 Perfect! Multi-sequence memory with no interference!")
        print("   ✅ Each sequence is independently stored")
        print("   ✅ Selective recall works correctly")
        print("   ✅ No cross-sequence activation")
    else:
        print(f"\n⚠️ {len(sequences) - success_count} sequence(s) failed.")

if __name__ == "__main__":
    run_multi_sequence_memory()

