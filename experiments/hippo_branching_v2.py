"""
================================================================================
HIPPO BRANCHING V2: True Parallel Activation (Associative Memory)
================================================================================

[v1 vs v2 차이]
v1 (CAT vs CAR):
- C→A→{T or R}  ← 경쟁 (Winner-Take-All)
- Basal Ganglia 스타일
- 하나만 선택

v2 (ANT, ARC, AIM):
- A→{N and R and I}  ← 병렬 (Parallel Activation)
- Hippocampus 스타일
- 여러 개 동시 활성화

[실험 목표]
Cue: "A"
Expected:
  t≈3ms: N, R, I 동시 발화 ✅
  t≈5ms: T, C, M 동시 발화 ✅
→ 3개 단어 모두 완성!

[핵심 차이]
- N, R, I는 서로 다른 뉴런 → 경쟁 안 함
- T, R은 같은 위치 → 경쟁함 (v1의 문제)
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
    print("🌳 HIPPOCAMPUS BRANCHING V2: Parallel Activation")
    print("=" * 70)
    print("Testing: A → {N, R, I} (simultaneous)")
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
    
    # ✅ 진짜 분기 실험: ANT, ARC, AIM (완전히 다른 경로!)
    words = {
        "ANT": {
            "letters": ["A", "N", "T"],
            "train_count": 10
        },
        "ARC": {
            "letters": ["A", "R", "C"],
            "train_count": 10
        },
        "AIM": {
            "letters": ["A", "I", "M"],
            "train_count": 10
        }
    }
    
    print(f"\n🌳 Branching scenario (True Parallel):")
    print(f"   ANT: A→N→T (train {words['ANT']['train_count']} times)")
    print(f"   ARC: A→R→C (train {words['ARC']['train_count']} times)")
    print(f"   AIM: A→I→M (train {words['AIM']['train_count']} times)")
    print(f"\n   Key difference from v1:")
    print(f"   - N, R, I are DIFFERENT neurons (no competition!)")
    print(f"   - All should fire simultaneously after 'A' cue")
    
    # 시냅스 생성
    word_synapses = {}
    total_synapses = []
    
    for word, config in words.items():
        letters = config["letters"]
        synapses = []
        
        for i in range(len(letters) - 1):
            letter1 = letters[i]
            letter2 = letters[i + 1]
            
            for pre_idx in letter_neurons[letter1]:
                for post_idx in letter_neurons[letter2]:
                    syn = STDPSynapse(neurons[pre_idx], neurons[post_idx], 
                                     delay_ms=2.0, Q_max=50.0)
                    neurons[pre_idx].outgoing_synapses.append(syn)
                    neurons[post_idx].incoming_synapses.append(syn)
                    synapses.append(syn)
                    total_synapses.append(syn)
        
        word_synapses[word] = synapses
    
    print(f"\n✅ Synapses created:")
    for word, syns in word_synapses.items():
        print(f"   {word}: {len(syns)} synapses")
    
    # =========================================================
    # PHASE 1: LEARNING (All words equally)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LEARNING (Equal frequency)")
    print("=" * 70)
    
    T_learn = 80.0
    steps = int(T_learn/dt)
    
    total_trains = sum(config["train_count"] for config in words.values())
    print(f"\nTotal training sessions: {total_trains}")
    
    train_session = 0
    for word, config in words.items():
        letters = config["letters"]
        train_count = config["train_count"]
        
        for rep in range(train_count):
            train_session += 1
            print(f"  [{train_session}/{total_trains}] Training '{word}'...", end="")
            
            for k in range(steps):
                t = k * dt
                I = np.zeros(N)
                
                # 순차적 자극
                for i, letter in enumerate(letters):
                    t_start = 5.0 + i * 15.0
                    t_end = t_start + 8.0
                    
                    if t_start < t < t_end:
                        for idx in letter_neurons[letter]:
                            I[idx] = 250.0 if i == 0 else 200.0
                
                # 뉴런 업데이트
                for i in range(N):
                    I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
                    neurons[i].step(dt, I[i] + I_syn_total, t)
                
                # 시냅스 전달
                for s in total_synapses:
                    s.deliver(t)
            
            # 세척 & Reset
            for _ in range(200):
                for i in range(N):
                    neurons[i].step(dt, 0.0, t)
                for s in total_synapses:
                    s.deliver(t)
            
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
            
            print(" Done.")
    
    # 가중치 확인
    print("\n🔍 Synaptic Weights After Learning:")
    a_neurons = letter_neurons["A"]
    
    for word in ["ANT", "ARC", "AIM"]:
        letters = words[word]["letters"]
        second_letter = letters[1]  # N, R, I
        
        weights = []
        for pre in a_neurons:
            for syn in neurons[pre].outgoing_synapses:
                if syn.post_neuron.name in [f"N{i}" for i in letter_neurons[second_letter]]:
                    weights.append(syn.weight)
        
        if weights:
            print(f"   A→{second_letter} ({word}): avg weight = {np.mean(weights):.2f}")
    
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
    for s in total_synapses:
        s.spikes = []
        s.I_syn = 0.0
    print("✅ Reset Done.")
    
    # =========================================================
    # PHASE 3: PARALLEL ACTIVATION TEST
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: PARALLEL ACTIVATION TEST")
    print("=" * 70)
    
    print("\n🧪 Critical Test: Cue 'A' → Will N, R, I fire TOGETHER?")
    
    T_test = 60.0
    steps_test = int(T_test/dt)
    
    cue = letter_neurons["A"]
    logs = []
    
    for k in range(steps_test):
        t = k * dt
        I = np.zeros(N)
        
        # Cue (1-5ms)
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
    
    # 분석: 각 문자의 발화 시간
    letter_first_spike = {}
    letter_spike_counts = {}
    
    for letter in ["A", "N", "R", "I", "T", "C", "M"]:
        letter_spike_counts[letter] = 0
        for t, ids in logs:
            if any(n in letter_neurons[letter] for n in ids):
                letter_spike_counts[letter] += 1
                if letter not in letter_first_spike:
                    letter_first_spike[letter] = t
    
    # 결과 출력
    print(f"\n📊 Activation Timeline:")
    for letter in ["A", "N", "R", "I", "T", "C", "M"]:
        count = letter_spike_counts[letter]
        first_t = letter_first_spike.get(letter, None)
        
        if first_t is not None:
            status = "✅" if count > 0 else "❌"
            print(f"   {letter}: {status} {count} spikes (First: {first_t:.1f}ms)")
        else:
            print(f"   {letter}: ❌ 0 spikes")
    
    # 병렬 활성화 판정
    print("\n🌳 Parallel Activation Analysis:")
    
    n_fired = letter_spike_counts["N"] > 0
    r_fired = letter_spike_counts["R"] > 0
    i_fired = letter_spike_counts["I"] > 0
    
    if n_fired and r_fired and i_fired:
        n_time = letter_first_spike["N"]
        r_time = letter_first_spike["R"]
        i_time = letter_first_spike["I"]
        
        time_diff = max(n_time, r_time, i_time) - min(n_time, r_time, i_time)
        
        print(f"   ✅ ALL three branches activated!")
        print(f"   N: {n_time:.1f}ms")
        print(f"   R: {r_time:.1f}ms")
        print(f"   I: {i_time:.1f}ms")
        print(f"   Time spread: {time_diff:.1f}ms")
        
        if time_diff < 2.0:
            print(f"\n   🎉 SIMULTANEOUS ACTIVATION! (Δt < 2ms)")
            print(f"   → True parallel branching confirmed!")
        else:
            print(f"\n   ✓ Sequential activation (Δt = {time_diff:.1f}ms)")
            print(f"   → All branches active, but slightly staggered")
    else:
        print(f"   ⚠️ Incomplete activation:")
        print(f"      N: {'✅' if n_fired else '❌'}")
        print(f"      R: {'✅' if r_fired else '❌'}")
        print(f"      I: {'✅' if i_fired else '❌'}")
    
    # 두 번째 레벨 (T, C, M) 확인
    t_fired = letter_spike_counts["T"] > 0
    c_fired = letter_spike_counts["C"] > 0
    m_fired = letter_spike_counts["M"] > 0
    
    print(f"\n🌿 Second Level Activation:")
    if t_fired and c_fired and m_fired:
        print(f"   ✅ ALL three endings activated!")
        print(f"   T: {letter_first_spike['T']:.1f}ms")
        print(f"   C: {letter_first_spike['C']:.1f}ms")
        print(f"   M: {letter_first_spike['M']:.1f}ms")
        print(f"\n   → Complete word formation: ANT, ARC, AIM")
    else:
        print(f"   T: {'✅' if t_fired else '❌'}")
        print(f"   C: {'✅' if c_fired else '❌'}")
        print(f"   M: {'✅' if m_fired else '❌'}")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    print("=" * 70)
    
    branches_active = sum([n_fired, r_fired, i_fired])
    completions_active = sum([t_fired, c_fired, m_fired])
    
    if branches_active == 3 and completions_active == 3:
        print("\n🎉 SUCCESS: PARALLEL BRANCHING CONFIRMED!")
        print(f"   ✅ All 3 branches activated: N, R, I")
        print(f"   ✅ All 3 completions: T, C, M")
        print(f"   ✅ Total words recalled: ANT, ARC, AIM")
        print(f"\n   → This is ASSOCIATIVE MEMORY!")
        print(f"   → Different from v1's Winner-Take-All")
    elif branches_active > 0:
        print(f"\n✓ PARTIAL SUCCESS:")
        print(f"   {branches_active}/3 branches activated")
        print(f"   {completions_active}/3 completions")
    else:
        print("\n❌ FAILED: No branching occurred")
    
    # =========================================================
    # VISUALIZATION
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 GENERATING VISUALIZATION...")
    print("=" * 70)
    
    fig = plt.figure(figsize=(16, 5))
    
    # 1. Activation Timeline
    ax1 = plt.subplot(1, 3, 1)
    letters = ["A", "N", "R", "I", "T", "C", "M"]
    times = [letter_first_spike.get(l, 0) for l in letters]
    colors_map = {"A": "#FFA07A", "N": "#FF6B6B", "R": "#4ECDC4", "I": "#98D8C8",
                  "T": "#FFD93D", "C": "#6BCB77", "M": "#4D96FF"}
    colors = [colors_map[l] for l in letters]
    
    ax1.bar(letters, times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('First Spike Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('[1] Activation Timeline', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Spike Counts
    ax2 = plt.subplot(1, 3, 2)
    counts = [letter_spike_counts[l] for l in letters]
    bars = ax2.bar(letters, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, counts):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Total Spikes', fontsize=12, fontweight='bold')
    ax2.set_title('[2] Spike Counts', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Branching Structure
    ax3 = plt.subplot(1, 3, 3)
    ax3.text(0.5, 0.9, 'A', ha='center', va='center', fontsize=20, fontweight='bold',
             bbox=dict(boxstyle='circle', facecolor='#FFA07A', edgecolor='black', linewidth=2))
    
    # First level
    for i, (letter, x) in enumerate([("N", 0.2), ("R", 0.5), ("I", 0.8)]):
        active = letter_spike_counts[letter] > 0
        color = colors_map[letter] if active else 'lightgray'
        ax3.text(x, 0.6, letter, ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor=color, edgecolor='black', linewidth=2))
        # Arrow
        ax3.annotate('', xy=(x, 0.65), xytext=(0.5, 0.85),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black' if active else 'gray'))
    
    # Second level
    for i, (letter, x) in enumerate([("T", 0.2), ("C", 0.5), ("M", 0.8)]):
        active = letter_spike_counts[letter] > 0
        color = colors_map[letter] if active else 'lightgray'
        ax3.text(x, 0.3, letter, ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor=color, edgecolor='black', linewidth=2))
        # Arrow
        parent = ["N", "R", "I"][i]
        ax3.annotate('', xy=(x, 0.35), xytext=(x, 0.55),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black' if active else 'gray'))
    
    # Labels
    ax3.text(0.2, 0.1, 'ANT', ha='center', fontsize=12, fontweight='bold')
    ax3.text(0.5, 0.1, 'ARC', ha='center', fontsize=12, fontweight='bold')
    ax3.text(0.8, 0.1, 'AIM', ha='center', fontsize=12, fontweight='bold')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('[3] Branching Structure', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = '/Users/jazzin/Desktop/hippo_v0/branching_v2_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {output_file}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✨ V1 vs V2 Comparison:")
    print("=" * 70)
    print("\n V1 (CAT vs CAR):")
    print("   - Winner-Take-All")
    print("   - T or R (exclusive)")
    print("   - Decision making")
    print("\n V2 (ANT, ARC, AIM):")
    print("   - Parallel Activation")
    print("   - N and R and I (inclusive)")
    print("   - Associative memory")
    print("\n → Both are correct, but serve different purposes! 🧠")

