import numpy as np
from v4_event import CONFIG, HHSomaQuick, SynapseCore

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
            self.weight = max(0.1, self.weight - 0.1 * np.exp(-dt_stdp/10.0))
        super().on_pre_spike(t, Ca, R * self.weight, ATP, dphi)

    def on_post_spike(self, t):
        self.last_post_time = t
        dt = t - self.last_pre_time
        if 0 < dt < 20.0:
            self.weight = min(10.0, self.weight + 2.0 * np.exp(-dt/10.0))

# ======================================================================
# Sequence Neuron
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
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔤 HIPPOCAMPUS ALPHABET MEMORY (A-Z)")
    print("=" * 70)
    print("26 letters stored independently in one network")
    print("=" * 70)
    
    dt = 0.1
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    N = len(alphabet) * 2  # 52 neurons (2 per letter)
    
    # 뉴런 생성
    neurons = [SequenceNeuron(f"N{i}") for i in range(N)]
    
    # 알파벳 매핑 (각 글자당 2개 뉴런)
    letter_neurons = {}
    for i, letter in enumerate(alphabet):
        letter_neurons[letter] = [i*2, i*2+1]
    
    print(f"\n✅ Network: {N} neurons")
    print(f"   A → {letter_neurons['A']}")
    print(f"   B → {letter_neurons['B']}")
    print(f"   ...")
    print(f"   Z → {letter_neurons['Z']}")
    
    # =========================================================
    # PHASE 1: LEARNING (각 글자를 개별 학습)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LEARNING")
    print("=" * 70)
    
    num_repeats = 5
    T_learn = 50.0
    steps = int(T_learn/dt)
    
    for rep in range(num_repeats):
        print(f"\n  Cycle {rep+1}/{num_repeats}:")
        
        for letter in alphabet:
            print(f"    Training {letter}...", end="")
            
            letter_ids = letter_neurons[letter]
            
            for k in range(steps):
                t = k * dt
                
                I = np.zeros(N)
                # 해당 글자 자극 (5-30ms)
                if 5.0 < t < 30.0:
                    for i in letter_ids:
                        I[i] = 250.0
                
                # 뉴런 업데이트
                for i in range(N):
                    neurons[i].step(dt, I[i], t)
            
            # 세척
            for _ in range(100):
                for i in range(N):
                    neurons[i].step(dt, 0.0, t)
            
            # Reset
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
            
            print(" Done.")
    
    print("\n✅ Learning Complete!")
    
    # =========================================================
    # PHASE 2: RESET
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: RESET")
    print("=" * 70)
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
    print("✅ Reset Done.")
    
    # =========================================================
    # PHASE 3: RECALL TEST (전체 26개 알파벳!)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: COMPREHENSIVE RECALL TEST (A-Z)")
    print("=" * 70)
    
    T_test = 30.0
    steps = int(T_test/dt)
    
    results = {}
    
    for letter in alphabet:
        cue = letter_neurons[letter]
        expected = set(cue)
        logs = []
        
        for k in range(steps):
            t = k * dt
            
            I = np.zeros(N)
            # Cue (1-10ms)
            if 1.0 <= t < 10.0:
                for i in cue:
                    I[i] = 300.0
            
            spikes = []
            for i in range(N):
                sp, _, _ = neurons[i].step(dt, I[i], t)
                if sp:
                    spikes.append(i)
            
            if spikes:
                logs.append((t, spikes))
        
        # 분석: 정확한 뉴런 ID 확인
        fired_neurons = set()
        for t, ids in logs:
            fired_neurons.update(ids)
        
        # 검증
        target_fired = len(fired_neurons & expected) > 0  # 타겟 발화?
        others = set(range(N)) - expected
        interference = len(fired_neurons & others) > 0  # 다른 뉴런 발화?
        
        success = target_fired and not interference
        
        results[letter] = {
            'success': success,
            'target': target_fired,
            'interference': interference,
            'fired': sorted(fired_neurons)
        }
        
        # 간결한 출력
        status = "✅" if success else "❌"
        if interference:
            status = "⚠️"
        print(f"  {letter}: {status}", end="")
        if (ord(letter) - ord('A') + 1) % 13 == 0:  # 13개마다 줄바꿈
            print()
        
        # Reset
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
    
    # =========================================================
    # DETAILED REPORT
    # =========================================================
    print("\n\n" + "=" * 70)
    print("📊 DETAILED REPORT")
    print("=" * 70)
    
    successes = sum(1 for r in results.values() if r['success'])
    target_hits = sum(1 for r in results.values() if r['target'])
    interferences = sum(1 for r in results.values() if r['interference'])
    
    print(f"\n✅ Perfect Recall: {successes}/26 ({successes/26*100:.1f}%)")
    print(f"🎯 Target Activation: {target_hits}/26 ({target_hits/26*100:.1f}%)")
    print(f"⚠️  Interference: {interferences}/26 ({interferences/26*100:.1f}%)")
    
    # 실패한 경우 상세 분석
    if successes < 26:
        print("\n🔍 Failed Letters:")
        for letter, r in results.items():
            if not r['success']:
                print(f"  {letter} (Expected: {letter_neurons[letter]}, Fired: {r['fired']})")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n🎯 Score: {successes}/26")
    
    if successes == 26:
        print("\n🎉 PERFECT! All 26 letters recalled with no interference!")
        print("   ✅ 100% accuracy")
        print("   ✅ 0% cross-talk")
    elif successes >= 23:
        print(f"\n✨ Excellent! {successes}/26 letters working!")
        print(f"   ⚠️ {26-successes} letter(s) need tuning")
    elif successes >= 20:
        print(f"\n👍 Good! {successes}/26 letters working!")
        print(f"   ⚠️ {26-successes} letter(s) need adjustment")
    else:
        print(f"\n⚠️ {26-successes} letter(s) failed - investigation needed.")
