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
    print("📖 HIPPOCAMPUS WORD MEMORY (Letter Sequences)")
    print("=" * 70)
    print("Learning words as sequences of letters")
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
    
    # ✅ 단어 정의 (알파벳 간 시냅스 연결)
    words = {
        "CAT": ["C", "A", "T"],
        "DOG": ["D", "O", "G"],
        "BAT": ["B", "A", "T"],
        "RAT": ["R", "A", "T"]
    }
    
    print(f"\n📖 Words to learn:")
    for word, letters in words.items():
        print(f"   {word} = {' → '.join(letters)}")
    
    # ✅ 단어별 시냅스 생성
    word_synapses = {}
    total_synapses = []
    
    for word, letters in words.items():
        synapses = []
        
        # 각 단어의 연속된 글자 간 연결
        for i in range(len(letters) - 1):
            letter1 = letters[i]
            letter2 = letters[i + 1]
            
            # letter1의 모든 뉴런 → letter2의 모든 뉴런
            for pre_idx in letter_neurons[letter1]:
                for post_idx in letter_neurons[letter2]:
                    syn = STDPSynapse(neurons[pre_idx], neurons[post_idx], delay_ms=2.0, Q_max=50.0)
                    neurons[pre_idx].outgoing_synapses.append(syn)
                    neurons[post_idx].incoming_synapses.append(syn)
                    synapses.append(syn)
                    total_synapses.append(syn)
        
        word_synapses[word] = synapses
    
    print(f"\n✅ Synapses created:")
    for word, syns in word_synapses.items():
        print(f"   {word}: {len(syns)} synapses")
    
    # =========================================================
    # PHASE 1: WORD LEARNING (단어 시퀀스 학습)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: WORD LEARNING (Sequential Patterns)")
    print("=" * 70)
    
    num_repeats = 10
    T_learn = 80.0
    steps = int(T_learn/dt)
    
    for rep in range(num_repeats):
        print(f"\n  Cycle {rep+1}/{num_repeats}:")
        
        for word, letters in words.items():
            print(f"    Training '{word}'...", end="")
            
            for k in range(steps):
                t = k * dt
                
                I = np.zeros(N)
                
                # 시간차 자극: 각 글자를 순차적으로
                for i, letter in enumerate(letters):
                    t_start = 5.0 + i * 15.0  # 0ms, 15ms, 30ms...
                    t_end = t_start + 8.0
                    
                    if t_start < t < t_end:
                        for idx in letter_neurons[letter]:
                            I[idx] = 250.0 if i == 0 else 200.0  # 첫 글자는 강하게
                
                # 뉴런 업데이트
                for i in range(N):
                    I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
                    neurons[i].step(dt, I[i] + I_syn_total, t)
                
                # 시냅스 전달
                for s in total_synapses:
                    s.deliver(t)
            
            # 세척
            for _ in range(200):
                for i in range(N):
                    neurons[i].step(dt, 0.0, t)
                for s in total_synapses:
                    s.deliver(t)
            
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
            for s in total_synapses:
                s.spikes = []
                s.I_syn = 0.0
                if hasattr(s, 'Ca'):
                    s.Ca = 0.0
                if hasattr(s, 'R'):
                    s.R = 1.0
            
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
    # PHASE 3: WORD RECALL TEST
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: WORD RECALL TEST")
    print("=" * 70)
    
    T_test = 60.0  # 단어는 더 긴 시간 필요
    steps = int(T_test/dt)
    
    results = {}
    
    for word, letters in words.items():
        print(f"\n🧪 Test: Cue '{word[0]}' → Expecting '{word}'")
        
        # 첫 글자만 Cue
        cue = letter_neurons[letters[0]]
        logs = []
        
        for k in range(steps):
            t = k * dt
            
            I = np.zeros(N)
            # Cue (1-5ms, 짧게)
            if 1.0 <= t < 5.0:
                for i in cue:
                    I[i] = 300.0
            
            spikes = []
            for i in range(N):
                I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
                sp, _, _ = neurons[i].step(dt, I[i] + I_syn_total, t)
                if sp:
                    spikes.append(i)
            
            # 시냅스 전달
            for s in total_synapses:
                s.deliver(t)
            
            if spikes:
                logs.append((t, spikes))
        
        # 분석: 각 글자가 순서대로 발화했는지 확인
        letter_activations = {}
        for t, ids in logs:
            for letter in letters:
                if any(n in letter_neurons[letter] for n in ids):
                    if letter not in letter_activations:
                        letter_activations[letter] = t
        
        # 검증
        all_fired = all(letter in letter_activations for letter in letters)
        
        if all_fired:
            # 순서 확인
            times = [letter_activations[letter] for letter in letters]
            correct_order = all(times[i] < times[i+1] for i in range(len(times)-1))
            
            sequence_str = " → ".join([f"{l}({letter_activations[l]:.1f}ms)" for l in letters])
            
            if correct_order:
                print(f"   ✅ {sequence_str}")
                results[word] = 'success'
            else:
                print(f"   ⚠️ Wrong order: {sequence_str}")
                results[word] = 'wrong_order'
        else:
            missing = [l for l in letters if l not in letter_activations]
            print(f"   ❌ Missing: {missing}")
            results[word] = 'incomplete'
        
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
        for s in total_synapses:
            s.spikes = []
            s.I_syn = 0.0
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    print("=" * 70)
    
    successes = sum(1 for r in results.values() if r == 'success')
    total_words = len(words)
    
    print(f"\n🎯 Score: {successes}/{total_words}")
    
    if successes == total_words:
        print("\n🎉 PERFECT! All words recalled in correct sequence!")
        print("   ✅ 100% accuracy")
        print("   ✅ Sequential order preserved")
    elif successes >= total_words * 0.75:
        print(f"\n✨ Good! {successes}/{total_words} words working!")
    else:
        print(f"\n⚠️ {total_words - successes} word(s) need adjustment.")
