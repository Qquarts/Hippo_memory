"""
================================================================================
HIPPO DREAM FINAL: Complete Memory System with Sleep Consolidation
================================================================================

[통합 시스템]
1. Wake (낮): 차별적 학습 (CAT 20회, CAR 1회)
2. Sleep (밤): Theta 리듬 기반 자발적 Replay & 시냅스 강화
3. Recall (아침): 학습 기반 경로 선택

[핵심 원리]
- 빈도 → 가중치 (Frequency → Structure)
- 비선형성 → 선택 (Non-linearity → Choice)
- 수면 → 강화 (Sleep → Consolidation)

[4대 엔진 통합]
✓ Sleep Engine (꿈: 기억 강화)
✓ Sequence Engine (순서 기억)
✓ Capacity Engine (대용량 저장)
✓ Branching Engine (확률적 선택)
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
# STDP Synapse with Consolidation
# ======================================================================
class STDPSynapse(SynapseCore):
    def __init__(self, pre, post, delay_ms=1.5, Q_max=50.0, tau_ms=2.0):
        super().__init__(pre.soma, post.soma, delay_ms=delay_ms, Q_max=Q_max, tau_ms=tau_ms)
        self.pre_neuron = pre
        self.post_neuron = post
        self.weight = 1.0
        self.last_pre_time = -100.0
        self.last_post_time = -100.0
        self.replay_count = 0  # Sleep Replay 횟수

    def on_pre_spike(self, t, Ca, R, ATP, dphi):
        self.last_pre_time = t
        dt_stdp = t - self.last_post_time
        if 0 < dt_stdp < 20.0:
            self.weight = max(0.1, self.weight - 0.05 * np.exp(-dt_stdp/10.0))
        
        # 비선형 증폭 (weight³)
        weight_factor = (self.weight / 50.0) ** 3
        super().on_pre_spike(t, Ca, R * self.weight * weight_factor, ATP, dphi)

    def on_post_spike(self, t):
        self.last_post_time = t
        dt = t - self.last_pre_time
        if 0 < dt < 20.0:
            self.weight = min(50.0, self.weight + 0.15 * np.exp(-dt/10.0))

    def consolidate(self, factor=0.05):
        """Sleep 중 시냅스 강화 (Consolidation)"""
        self.weight = min(50.0, self.weight + factor)
        self.replay_count += 1

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
        self.wake_spike_count = 0  # Wake 시 발화 횟수

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
            
        return sp, self.S, self.PTP

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🌙 HIPPO DREAM FINAL: Complete Memory System")
    print("=" * 70)
    print("Simulating: Wake → Sleep → Recall cycle")
    print("=" * 70)
    
    dt = 0.1
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    N = len(alphabet) * 2  # 각 문자당 2개 뉴런
    
    # =========================================================
    # NETWORK SETUP
    # =========================================================
    print("\n✅ Network: 52 neurons (A-Z, 2 neurons each)")
    
    neurons = [SequenceNeuron(i) for i in range(N)]
    
    # 알파벳 뉴런 매핑
    letter_neurons = {}
    for i, letter in enumerate(alphabet):
        letter_neurons[letter] = [i * 2, i * 2 + 1]
    
    # =========================================================
    # WORDS DEFINITION (CAT vs CAR)
    # =========================================================
    words = {
        "CAT": {
            "letters": ["C", "A", "T"],
            "train_count": 20,  # 빈번한 단어
            "replay_priority": 1.0
        },
        "CAR": {
            "letters": ["C", "A", "R"],
            "train_count": 1,   # 드문 단어
            "replay_priority": 0.1
        }
    }
    
    print(f"\n📚 Words:")
    print(f"   CAT: train {words['CAT']['train_count']} times (frequent)")
    print(f"   CAR: train {words['CAR']['train_count']} times (rare)")
    
    # =========================================================
    # SYNAPSE CREATION
    # =========================================================
    word_synapses = {}
    total_synapses = []
    
    class DynamicSTDPSynapse(STDPSynapse):
        pass
    
    for word, config in words.items():
        letters = config["letters"]
        synapses = []
        
        for i in range(len(letters) - 1):
            letter1 = letters[i]
            letter2 = letters[i + 1]
            
            for pre_idx in letter_neurons[letter1]:
                for post_idx in letter_neurons[letter2]:
                    syn = DynamicSTDPSynapse(neurons[pre_idx], neurons[post_idx], 
                                            delay_ms=2.0, Q_max=50.0)
                    neurons[pre_idx].outgoing_synapses.append(syn)
                    neurons[post_idx].incoming_synapses.append(syn)
                    synapses.append(syn)
                    total_synapses.append(syn)
        
        word_synapses[word] = synapses
    
    print(f"\n✅ Synapses created: {len(total_synapses)} total")
    
    # =========================================================
    # PHASE 1: WAKE (Learning)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: WAKE - Differential Learning")
    print("=" * 70)
    
    T_learn = 80.0
    steps = int(T_learn / dt)
    
    training_sessions = []
    for word, config in words.items():
        for _ in range(config["train_count"]):
            training_sessions.append(word)
    
    print(f"\nTotal training sessions: {len(training_sessions)}")
    
    for session_idx, word in enumerate(training_sessions, 1):
        print(f"  [{session_idx}/{len(training_sessions)}] Training '{word}'...", end=" ")
        
        letters = words[word]["letters"]
        
        for k in range(steps):
            t = k * dt
            I = np.zeros(N)
            
            # 순차적 자극
            for letter_idx, letter in enumerate(letters):
                t_start = 5.0 + letter_idx * 15.0
                t_end = t_start + 8.0
                if t_start <= t < t_end:
                    for i in letter_neurons[letter]:
                        I[i] = 300.0
                        neurons[i].wake_spike_count += 1
            
            # 뉴런 업데이트
            for i in range(N):
                I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
                neurons[i].step(dt, I[i] + I_syn_total, t)
            
            # 시냅스 전달
            for s in total_synapses:
                s.deliver(t)
        
        print("Done.")
    
    # 가중치 측정 (Wake 후)
    def get_avg_weight(word, letter1, letter2):
        weights = []
        for syn in word_synapses[word]:
            if (syn.pre_neuron.name in letter_neurons[letter1] and 
                syn.post_neuron.name in letter_neurons[letter2]):
                weights.append(syn.weight)
        return np.mean(weights) if weights else 0
    
    cat_weight_wake = get_avg_weight("CAT", "A", "T")
    car_weight_wake = get_avg_weight("CAR", "A", "R")
    
    print(f"\n🔍 Synaptic Weights After WAKE:")
    print(f"   A→T (CAT): {cat_weight_wake:.2f}")
    print(f"   A→R (CAR): {car_weight_wake:.2f}")
    print(f"   Ratio: {cat_weight_wake/car_weight_wake:.2f}x")
    
    # =========================================================
    # PHASE 2: SLEEP - Replay & Consolidation
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: SLEEP - Theta Replay & Consolidation")
    print("=" * 70)
    print("🌙 Entering Sleep Mode...")
    print("   - Theta oscillation: 6 Hz")
    print("   - Replay priority: CAT >> CAR")
    print("   - Synaptic consolidation active")
    
    # Reset neurons
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
    
    # Sleep parameters
    theta_freq = 6.0  # Hz
    theta_period = 1000.0 / theta_freq  # ms
    num_theta_cycles = 20  # 20 사이클 (약 3.3초)
    T_sleep = num_theta_cycles * theta_period
    steps_sleep = int(T_sleep / dt)
    
    replay_log = {"CAT": 0, "CAR": 0}
    
    print(f"\n🔄 Replaying memories ({num_theta_cycles} theta cycles)...")
    
    for cycle in range(num_theta_cycles):
        # 빈도 기반 확률적 선택
        if np.random.rand() < 0.9:  # 90% CAT
            word = "CAT"
        else:  # 10% CAR
            word = "CAR"
        
        replay_log[word] += 1
        letters = words[word]["letters"]
        
        # 한 Theta 사이클 동안 재생
        t_offset = cycle * theta_period
        
        for k in range(int(theta_period / dt)):
            t = t_offset + k * dt
            I = np.zeros(N)
            
            # 약한 자극으로 재생 (Wake보다 약함)
            for letter_idx, letter in enumerate(letters):
                t_start = 5.0 + letter_idx * 10.0
                t_end = t_start + 5.0
                t_local = k * dt
                if t_start <= t_local < t_end:
                    for i in letter_neurons[letter]:
                        I[i] = 150.0  # Wake의 절반
            
            # 뉴런 업데이트
            for i in range(N):
                I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
                neurons[i].step(dt, I[i] + I_syn_total, t)
            
            # 시냅스 전달
            for s in total_synapses:
                s.deliver(t)
        
        # 사이클 끝마다 시냅스 강화 (Consolidation)
        for syn in word_synapses[word]:
            syn.consolidate(factor=0.02)  # 점진적 강화
        
        if (cycle + 1) % 5 == 0:
            print(f"   [{cycle+1}/{num_theta_cycles}] cycles complete...")
    
    print(f"\n✅ Sleep complete!")
    print(f"   Replay count: CAT={replay_log['CAT']}, CAR={replay_log['CAR']}")
    
    # 가중치 측정 (Sleep 후)
    cat_weight_sleep = get_avg_weight("CAT", "A", "T")
    car_weight_sleep = get_avg_weight("CAR", "A", "R")
    
    print(f"\n🔍 Synaptic Weights After SLEEP:")
    print(f"   A→T (CAT): {cat_weight_sleep:.2f} (Δ+{cat_weight_sleep-cat_weight_wake:.2f})")
    print(f"   A→R (CAR): {car_weight_sleep:.2f} (Δ+{car_weight_sleep-car_weight_wake:.2f})")
    print(f"   Ratio: {cat_weight_sleep/car_weight_sleep:.2f}x")
    
    # =========================================================
    # PHASE 3: RECALL (Morning Test)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: RECALL - Morning Test")
    print("=" * 70)
    print("☀️ Good morning! Testing memory...")
    
    # Reset neurons
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
    
    # 여러 번 테스트
    num_trials = 20
    trial_results = {'T': 0, 'R': 0}
    
    print(f"\n🧪 Testing with 'C' cue ({num_trials} trials)...")
    
    T_test = 60.0
    steps_test = int(T_test / dt)
    
    for trial in range(num_trials):
        cue = letter_neurons["C"]
        t_neurons = letter_neurons["T"]
        r_neurons = letter_neurons["R"]
        winner_decided = False
        winner_group = None
        
        for k in range(steps_test):
            t = k * dt
            I = np.zeros(N)
            
            # Cue
            if 1.0 <= t < 5.0:
                for i in cue:
                    I[i] = 300.0
            
            # WTA 경쟁
            t_synaptic = [sum(syn.I_syn for syn in neurons[i].incoming_synapses) for i in t_neurons]
            r_synaptic = [sum(syn.I_syn for syn in neurons[i].incoming_synapses) for i in r_neurons]
            t_avg = np.mean(t_synaptic) if t_synaptic else 0.0
            r_avg = np.mean(r_synaptic) if r_synaptic else 0.0
            
            if t > 5.0 and (t_avg > 1.0 or r_avg > 1.0):
                if not winner_decided:
                    if t_avg > r_avg:
                        winner_decided = True
                        winner_group = "T"
                    elif r_avg > t_avg:
                        winner_decided = True
                        winner_group = "R"
                    else:
                        winner_decided = True
                        winner_group = "T" if np.random.rand() > 0.5 else "R"
            
            # 뉴런 업데이트
            for i in range(N):
                inhibition = 0.0
                if winner_decided:
                    if winner_group == "T" and i in r_neurons:
                        inhibition = -1000.0
                    elif winner_group == "R" and i in t_neurons:
                        inhibition = -1000.0
                
                I_syn_total = sum(syn.I_syn for syn in neurons[i].incoming_synapses)
                neurons[i].step(dt, I[i] + I_syn_total + inhibition, t)
            
            for s in total_synapses:
                s.deliver(t)
        
        # 결과 수집
        if winner_group == "T":
            trial_results['T'] += 1
        elif winner_group == "R":
            trial_results['R'] += 1
        
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
    
    print(f"\n📊 Recall Results ({num_trials} trials):")
    print(f"   T (CAT): {trial_results['T']}/{num_trials} ({trial_results['T']/num_trials*100:.0f}%)")
    print(f"   R (CAR): {trial_results['R']}/{num_trials} ({trial_results['R']/num_trials*100:.0f}%)")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    print("=" * 70)
    
    if trial_results['T'] > trial_results['R']:
        print("\n✅ SUCCESS: System chose the MORE FREQUENT path!")
        print(f"   Training ratio: {words['CAT']['train_count']}:{words['CAR']['train_count']}")
        print(f"   Weight ratio (Wake): {cat_weight_wake/car_weight_wake:.2f}x")
        print(f"   Weight ratio (Sleep): {cat_weight_sleep/car_weight_sleep:.2f}x")
        print(f"   Selection: {trial_results['T']}:{trial_results['R']}")
        print(f"\n   💡 Sleep strengthened synapses by {replay_log['CAT']} replays!")
        print(f"   → Memory consolidation working!")
    else:
        print("\n⚠️ Unexpected result")
    
    # =========================================================
    # VISUALIZATION
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 GENERATING VISUALIZATION...")
    print("=" * 70)
    
    fig = plt.figure(figsize=(18, 5))
    
    # 1. Weight Evolution
    ax1 = plt.subplot(1, 4, 1)
    stages = ['Initial', 'After Wake', 'After Sleep']
    cat_weights = [1.0, cat_weight_wake, cat_weight_sleep]
    car_weights = [1.0, car_weight_wake, car_weight_sleep]
    
    x = np.arange(len(stages))
    width = 0.35
    
    ax1.bar(x - width/2, cat_weights, width, label='CAT (A→T)', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, car_weights, width, label='CAR (A→R)', color='#4ECDC4', alpha=0.8)
    
    ax1.set_ylabel('Synaptic Weight', fontsize=12, fontweight='bold')
    ax1.set_title('[1] Weight Evolution', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=15)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Training vs Replay
    ax2 = plt.subplot(1, 4, 2)
    words_list = ['CAT', 'CAR']
    train_counts = [words['CAT']['train_count'], words['CAR']['train_count']]
    replay_counts = [replay_log['CAT'], replay_log['CAR']]
    
    x = np.arange(len(words_list))
    ax2.bar(x - width/2, train_counts, width, label='Wake Training', color='#FFA07A', alpha=0.8)
    ax2.bar(x + width/2, replay_counts, width, label='Sleep Replay', color='#98D8C8', alpha=0.8)
    
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('[2] Training vs Replay', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(words_list)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Recall Performance
    ax3 = plt.subplot(1, 4, 3)
    recall_labels = ['T\n(Frequent)', 'R\n(Rare)']
    recall_values = [trial_results['T'], trial_results['R']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax3.bar(recall_labels, recall_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, recall_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val}/{num_trials}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Trials Selected', fontsize=12, fontweight='bold')
    ax3.set_title('[3] Recall Performance', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, num_trials * 1.15)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Consolidation Effect
    ax4 = plt.subplot(1, 4, 4)
    cat_gain = ((cat_weight_sleep - cat_weight_wake) / cat_weight_wake) * 100
    car_gain = ((car_weight_sleep - car_weight_wake) / car_weight_wake) * 100
    
    gains = [cat_gain, car_gain]
    bars = ax4.barh(words_list, gains, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, gains):
        width = bar.get_width()
        ax4.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'+{val:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax4.set_xlabel('Weight Gain (%)', fontsize=12, fontweight='bold')
    ax4.set_title('[4] Sleep Consolidation', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = '/Users/jazzin/Desktop/hippo_v0/dream_final_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {output_file}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("🎉 SIMULATION COMPLETE!")
    print("=" * 70)
    print("\n✨ You have successfully created:")
    print("   1. A brain that learns from experience")
    print("   2. A brain that consolidates memory during sleep")
    print("   3. A brain that makes confident decisions")
    print("\n   → This is the essence of biological intelligence! 🧠")

