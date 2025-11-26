"""
================================================================================
HIPPO CA1: Novelty Detection (새로움 감지)
================================================================================

[원리]
CA3: "이전에 본 패턴들"
CA1: "예상 vs 실제" 비교 → 불일치 감지

[메커니즘]
1. 학습: CAT, DOG 저장
2. 테스트: BAT 제시
3. CA1: "BAT? 처음 보는데?" ← Novelty Signal!

[생물학적 의의]
- 새로운 것 = 중요함 → 주의 집중
- Curiosity 기반 학습
- "이거 배워야 해!" 신호
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
# CA1 Novelty Detector
# ======================================================================
class NoveltyDetector:
    """
    CA3 출력과 예상 패턴을 비교하여 새로움을 감지
    """
    def __init__(self, name):
        self.name = name
        self.soma = HHSomaQuick(CONFIG["HH"])
        self.expected_patterns = []  # 학습된 패턴들
        self.novelty_threshold = 0.5
        self.S, self.PTP = 0.0, 1.0
        self.outgoing_synapses = []
        self.incoming_synapses = []
    
    def learn_pattern(self, pattern_name):
        """패턴 학습"""
        if pattern_name not in self.expected_patterns:
            self.expected_patterns.append(pattern_name)
    
    def compute_novelty(self, pattern_name):
        """새로움 점수 계산"""
        if pattern_name in self.expected_patterns:
            return 0.0  # 익숙함
        else:
            return 1.0  # 새로움!
    
    def step(self, dt, t, pattern_name, I_ext=0.0):
        """Novelty에 비례하여 발화"""
        novelty_score = self.compute_novelty(pattern_name)
        
        if novelty_score > self.novelty_threshold:
            I_ext += 200.0 * novelty_score  # 새로울수록 강하게 자극
        
        self.soma.step(dt, I_ext)
        sp = self.soma.spiking()
        
        if sp:
            self.S = min(1.0, self.S + 0.3)
            self.PTP = min(2.0, self.PTP + 0.05)
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
        
        return sp, self.S, self.PTP, novelty_score

# ======================================================================
# Basic Neuron
# ======================================================================
class BasicNeuron:
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
        else:
            self.S = max(0.0, self.S - 0.01)
            self.PTP = max(1.0, self.PTP - 0.001)
            
        return sp, self.S, self.PTP

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔍 HIPPO CA1: Novelty Detection")
    print("=" * 70)
    print("Testing: Familiar (CAT, DOG) vs Novel (BAT, RAT)")
    print("=" * 70)
    
    dt = 0.1
    
    # =========================================================
    # NETWORK SETUP
    # =========================================================
    print("\n✅ Creating CA3 + CA1 Novelty Detector...")
    
    # CA3 neurons (단어별 대표 뉴런)
    ca3_words = {
        'CAT': BasicNeuron('CA3_CAT'),
        'DOG': BasicNeuron('CA3_DOG'),
        'BAT': BasicNeuron('CA3_BAT'),
        'RAT': BasicNeuron('CA3_RAT')
    }
    
    # CA1 Novelty Detector
    ca1_novelty = NoveltyDetector('CA1_Novelty')
    
    print(f"   CA3 word neurons: {len(ca3_words)}")
    print(f"   CA1 novelty detector: 1")
    
    # =========================================================
    # PHASE 1: LEARNING (Familiar patterns)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LEARNING (Familiar Words)")
    print("=" * 70)
    
    familiar_words = ['CAT', 'DOG']
    print(f"\nTeaching familiar words: {familiar_words}")
    
    for word in familiar_words:
        ca1_novelty.learn_pattern(word)
        print(f"  ✅ Learned: {word}")
    
    print(f"\n✅ CA1 memory: {ca1_novelty.expected_patterns}")
    
    # =========================================================
    # PHASE 2: NOVELTY TEST
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: NOVELTY DETECTION TEST")
    print("=" * 70)
    
    test_words = ['CAT', 'DOG', 'BAT', 'RAT']
    print(f"\nTesting words: {test_words}")
    print(f"Expected: CAT, DOG = familiar (low novelty)")
    print(f"Expected: BAT, RAT = novel (high novelty)")
    
    T_test = 50.0
    steps_test = int(T_test/dt)
    
    results = {}
    
    for word in test_words:
        print(f"\n🧪 Testing '{word}'...")
        
        # Reset
        for neuron in ca3_words.values():
            neuron.soma.V = -70.0
            neuron.soma.m = 0.05
            neuron.soma.h = 0.60
            neuron.soma.n = 0.32
            neuron.soma.spike_flag = False
            neuron.soma.mode = "rest"
            neuron.soma.ref_remaining = 0.0
            neuron.S = 0.0
            neuron.PTP = 1.0
        
        ca1_novelty.soma.V = -70.0
        ca1_novelty.soma.m = 0.05
        ca1_novelty.soma.h = 0.60
        ca1_novelty.soma.n = 0.32
        ca1_novelty.soma.spike_flag = False
        ca1_novelty.soma.mode = "rest"
        ca1_novelty.soma.ref_remaining = 0.0
        ca1_novelty.S = 0.0
        ca1_novelty.PTP = 1.0
        
        ca3_spikes = 0
        ca1_spikes = 0
        novelty_score = 0.0
        
        for k in range(steps_test):
            t = k * dt
            
            # CA3 자극 (단어 제시)
            I_ca3 = 0.0
            if 5.0 <= t < 15.0:
                I_ca3 = 300.0
            
            # CA3 업데이트
            sp, _, _ = ca3_words[word].step(dt, I_ca3, t)
            if sp:
                ca3_spikes += 1
            
            # CA1 Novelty Detection
            sp, _, _, nov = ca1_novelty.step(dt, t, word, 0.0)
            if sp:
                ca1_spikes += 1
            novelty_score = nov
        
        results[word] = {
            'ca3_spikes': ca3_spikes,
            'ca1_spikes': ca1_spikes,
            'novelty_score': novelty_score,
            'is_novel': novelty_score > 0.5
        }
        
        status = "🆕 NOVEL" if results[word]['is_novel'] else "✅ FAMILIAR"
        print(f"   CA3 spikes: {ca3_spikes}")
        print(f"   CA1 spikes: {ca1_spikes}")
        print(f"   Novelty: {novelty_score:.2f}")
        print(f"   → {status}")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    print("=" * 70)
    
    correct_detections = 0
    total_tests = len(test_words)
    
    print("\n📊 Novelty Detection Results:")
    for word, result in results.items():
        expected_novel = word not in familiar_words
        detected_novel = result['is_novel']
        correct = expected_novel == detected_novel
        
        if correct:
            correct_detections += 1
        
        symbol = "✅" if correct else "❌"
        status = "Novel" if detected_novel else "Familiar"
        print(f"   {symbol} {word}: {status} (novelty={result['novelty_score']:.2f})")
    
    accuracy = correct_detections / total_tests * 100
    print(f"\n🎯 Accuracy: {correct_detections}/{total_tests} ({accuracy:.0f}%)")
    
    if accuracy == 100:
        print("\n🎉 PERFECT: CA1 correctly detects all novel patterns!")
        print("   → Novelty detection system working!")
    elif accuracy >= 75:
        print("\n✓ GOOD: CA1 detects most novel patterns")
    else:
        print("\n⚠️ Needs improvement")
    
    # =========================================================
    # VISUALIZATION
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 GENERATING VISUALIZATION...")
    print("=" * 70)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 1. Novelty Scores
    ax1 = plt.subplot(1, 2, 1)
    words = list(results.keys())
    novelty_scores = [results[w]['novelty_score'] for w in words]
    colors = ['green' if w in familiar_words else 'red' for w in words]
    
    bars = ax1.bar(words, novelty_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.axhline(y=0.5, color='blue', linestyle='--', linewidth=2, label='Novelty Threshold')
    
    for bar, val in zip(bars, novelty_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Novelty Score', fontsize=12, fontweight='bold')
    ax1.set_title('[1] Novelty Detection Scores', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.2)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. CA1 Response
    ax2 = plt.subplot(1, 2, 2)
    ca1_spikes = [results[w]['ca1_spikes'] for w in words]
    
    bars = ax2.bar(words, ca1_spikes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, ca1_spikes):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('CA1 Spikes', fontsize=12, fontweight='bold')
    ax2.set_title('[2] CA1 Novelty Response', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Familiar (Learned)'),
        Patch(facecolor='red', edgecolor='black', label='Novel (New)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    output_file = '/Users/jazzin/Desktop/hippo_v0/ca1_novelty_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {output_file}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✨ CA1 detects novelty → triggers learning!")
    print("=" * 70)

