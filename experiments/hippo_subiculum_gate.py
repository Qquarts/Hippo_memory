"""
================================================================================
HIPPO SUBICULUM: Context-Based Output Gating
================================================================================

[원리]
CA3: 여러 기억 동시 활성화 (ANT, ARC, AIM)
CA1: 시간/새로움 정보 추가
Subiculum: 맥락에 맞는 것만 출력 ← 최종 제어!

[메커니즘]
1. Context Signal 입력 ("곤충 이야기 중")
2. CA3 출력 필터링
3. 맥락에 맞는 것만 강화, 나머지 억제
4. 깔끔한 출력 생성

[생물학적 의의]
- 상황에 맞는 기억만 떠올림
- 불필요한 정보 억제
- 효율적인 의사소통
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
# Subiculum Gate
# ======================================================================
class SubiculumGate:
    """
    맥락 기반 출력 제어
    """
    def __init__(self, name):
        self.name = name
        self.soma = HHSomaQuick(CONFIG["HH"])
        self.context_memory = {}  # {"insect": ["ANT"], "vehicle": ["CAR"], ...}
        self.current_context = None
        self.S, self.PTP = 0.0, 1.0
        self.outgoing_synapses = []
        self.incoming_synapses = []
    
    def set_context(self, context):
        """맥락 설정"""
        self.current_context = context
    
    def learn_context_association(self, context, word):
        """맥락-단어 연관 학습"""
        if context not in self.context_memory:
            self.context_memory[context] = []
        if word not in self.context_memory[context]:
            self.context_memory[context].append(word)
    
    def compute_relevance(self, word):
        """맥락 관련성 점수"""
        if self.current_context is None:
            return 0.5  # 맥락 없으면 중립
        
        if self.current_context in self.context_memory:
            relevant_words = self.context_memory[self.current_context]
            if word in relevant_words:
                return 1.0  # 맥락과 일치!
            else:
                return 0.0  # 맥락과 불일치
        
        return 0.5  # 모르는 맥락
    
    def gate(self, word, ca_input):
        """출력 게이팅"""
        relevance = self.compute_relevance(word)
        return ca_input * relevance  # 관련성에 비례하여 통과

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
    print("🚪 HIPPO SUBICULUM: Context-Based Gating")
    print("=" * 70)
    print("Testing: ANT, ARC, AIM → Context filtering")
    print("=" * 70)
    
    dt = 0.1
    
    # =========================================================
    # NETWORK SETUP
    # =========================================================
    print("\n✅ Creating CA3 + Subiculum network...")
    
    # CA3 neurons (각 단어별)
    ca3_words = {
        'ANT': BasicNeuron('CA3_ANT'),
        'ARC': BasicNeuron('CA3_ARC'),
        'AIM': BasicNeuron('CA3_AIM')
    }
    
    # Subiculum gates (각 단어별 출력 게이트)
    subiculum_gates = {
        'ANT': SubiculumGate('Sub_ANT'),
        'ARC': SubiculumGate('Sub_ARC'),
        'AIM': SubiculumGate('Sub_AIM')
    }
    
    print(f"   CA3 word neurons: {len(ca3_words)}")
    print(f"   Subiculum gates: {len(subiculum_gates)}")
    
    # =========================================================
    # PHASE 1: CONTEXT LEARNING
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: CONTEXT-WORD ASSOCIATION LEARNING")
    print("=" * 70)
    
    # 맥락-단어 연관 학습
    context_associations = {
        "insect": ["ANT"],
        "shape": ["ARC"],
        "action": ["AIM"]
    }
    
    print("\nTeaching context associations:")
    for context, words in context_associations.items():
        for word in words:
            subiculum_gates[word].learn_context_association(context, word)
            print(f"  ✅ {context} → {word}")
    
    print("\n✅ Subiculum context memory:")
    for word, gate in subiculum_gates.items():
        print(f"   {word}: {gate.context_memory}")
    
    # =========================================================
    # PHASE 2: GATING TEST
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: CONTEXT-BASED GATING TEST")
    print("=" * 70)
    
    test_contexts = ["insect", "shape", "action", None]
    T_test = 50.0
    steps_test = int(T_test/dt)
    
    all_results = {}
    
    for context in test_contexts:
        context_name = context if context else "no_context"
        print(f"\n🎯 Testing with context: '{context_name}'")
        
        # 맥락 설정
        for gate in subiculum_gates.values():
            gate.set_context(context)
        
        # 모든 CA3 동시 활성화 (병렬 분기)
        results = {}
        
        for word in ca3_words.keys():
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
            
            ca3_spikes = 0
            sub_output = 0.0
            
            for k in range(steps_test):
                t = k * dt
                
                # CA3 자극 (단어 활성화)
                I_ca3 = 0.0
                if 5.0 <= t < 15.0:
                    I_ca3 = 300.0
                
                # CA3 업데이트
                sp, _, _ = ca3_words[word].step(dt, I_ca3, t)
                if sp:
                    ca3_spikes += 1
                    # Subiculum gate 통과
                    sub_output += subiculum_gates[word].gate(word, 1.0)
            
            relevance = subiculum_gates[word].compute_relevance(word)
            results[word] = {
                'ca3_spikes': ca3_spikes,
                'sub_output': sub_output,
                'relevance': relevance
            }
        
        all_results[context_name] = results
        
        # 결과 출력
        print(f"\n  CA3 Output (all active):")
        for word, result in results.items():
            print(f"    {word}: {result['ca3_spikes']} spikes")
        
        print(f"\n  Subiculum Output (filtered):")
        for word, result in results.items():
            relevance = result['relevance']
            output = result['sub_output']
            
            if relevance > 0.7:
                status = "✅ PASS (relevant)"
            elif relevance < 0.3:
                status = "❌ BLOCK (irrelevant)"
            else:
                status = "⚠️  NEUTRAL"
            
            print(f"    {word}: relevance={relevance:.2f}, output={output:.1f} → {status}")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY")
    print("=" * 70)
    
    print("\n📊 Context-Based Filtering:")
    for context_name, results in all_results.items():
        print(f"\n  Context: '{context_name}'")
        
        # 가장 높은 relevance 찾기
        max_relevance = max(r['relevance'] for r in results.values())
        selected_words = [w for w, r in results.items() if r['relevance'] == max_relevance and max_relevance > 0.5]
        
        if selected_words:
            print(f"    → Selected: {', '.join(selected_words)} ✅")
        else:
            print(f"    → No clear selection (neutral context)")
    
    # 정확도 계산
    expected_selections = {
        "insect": "ANT",
        "shape": "ARC",
        "action": "AIM"
    }
    
    correct_selections = 0
    total_tests = len(expected_selections)
    
    for context, expected in expected_selections.items():
        results = all_results[context]
        selected = max(results.items(), key=lambda x: x[1]['relevance'])[0]
        
        if selected == expected:
            correct_selections += 1
    
    accuracy = correct_selections / total_tests * 100
    print(f"\n🎯 Gating Accuracy: {correct_selections}/{total_tests} ({accuracy:.0f}%)")
    
    if accuracy == 100:
        print("\n🎉 PERFECT: Subiculum correctly gates based on context!")
        print("   → Context-based output control working!")
    elif accuracy >= 67:
        print("\n✓ GOOD: Most contexts correctly gated")
    else:
        print("\n⚠️ Needs improvement")
    
    # =========================================================
    # VISUALIZATION
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 GENERATING VISUALIZATION...")
    print("=" * 70)
    
    fig = plt.figure(figsize=(16, 5))
    
    # 각 맥락별 그래프
    contexts_to_plot = ["insect", "shape", "action"]
    
    for idx, context in enumerate(contexts_to_plot, 1):
        ax = plt.subplot(1, 3, idx)
        
        results = all_results[context]
        words = list(results.keys())
        relevances = [results[w]['relevance'] for w in words]
        
        colors = ['green' if r > 0.7 else 'red' if r < 0.3 else 'gray' for r in relevances]
        
        bars = ax.bar(words, relevances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Pass')
        ax.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Block')
        
        for bar, val in zip(bars, relevances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Relevance', fontsize=11, fontweight='bold')
        ax.set_title(f'Context: "{context}"', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = '/Users/jazzin/Desktop/hippo_v0/subiculum_gate_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {output_file}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✨ Subiculum filters output based on context!")
    print("=" * 70)

