"""
===================================================================================
HIPPO_DREAM_v1 — Hippocampus-Cortex Memory Consolidation System
===================================================================================
지은이: GNJz | 발행: 2025.11.24

Wake → Sleep → Wake 메모리 공고화 시스템

📐 구현된 핵심 수식:

1️⃣ Hodgkin-Huxley Neuron Dynamics (HHSomaQuick):
   C_m dV/dt = I_ext + I_syn - g_L(V-E_L) - g_Na·m³h(V-E_Na) - g_K·n⁴(V-E_K)
   
   Gating variables:
   dm/dt = α_m(1-m) - β_m·m
   dh/dt = α_h(1-h) - β_h·h  
   dn/dt = α_n(1-n) - β_n·n
   
   Spike condition: V > spike_thresh (=0.0 mV)

2️⃣ Short-Term Plasticity (STP) & Post-Tetanic Potentiation (PTP):
   On spike:  S ← S + 0.3,    PTP ← PTP + 0.05
   Decay:     S ← S - 0.01,   PTP ← PTP - 0.001

3️⃣ Subiculum Integration (Low-pass filter):
   y(t+dt) = (1-α)·y(t) + spike(t)
   where α = dt/τ

4️⃣ Cortex Ridge Regression (Initial learning):
   W = Y·X^T·(X·X^T + αI)^(-1)
   Inference: p_i = exp(z_i) / Σ_j exp(z_j)  (Softmax)

5️⃣ Cortex Incremental Learning (During sleep):
   error = y - ŷ
   W ← W + η·(error ⊗ input)

6️⃣ Hippocampal Replay (Sleep phase):
   I_DG = I_base + N(0, σ)  (Weak input + noise)
   Q_ij ← Q_ij + f(S, PTP)  (Synaptic reinforcement)

7️⃣ Hippocampal Synaptic Decay:
   Q_max ← Q_max · decay_rate

8️⃣ Network Architecture:
   Wake:  DG → CA3 (clusters) → CA1 → Subiculum → Cortex
   Sleep: Replay → Consolidation → Decay
   Wake:  Hippocampus test → Cortex direct test

===================================================================================
"""

"""
===================================================================================
📦 Dependencies
===================================================================================

This module depends on `v3_event.py` which contains:
  - CONFIG (global Hodgkin-Huxley parameters)
  - HHSomaQuick (fast HH soma)
  - SynapseCore (synaptic event engine with delay queue)

If v3_event.py is not available, please check the project repository
or contact the author for the full package.

===================================================================================
"""

# Qquarts co Present
# 지은이 : GNJz 
# 발행 2025.11.24

import numpy as np
import random

# ✅ 핵심 엔진 임포트
from v3_event import CONFIG, HHSomaQuick, SynapseCore

# ======================================================================
# 1. Configuration
# ======================================================================

# 🎲 Reproducibility: 모든 난수 생성기를 동일 시드로 고정
random.seed(42)
np.random.seed(42)

# 📐 Hodgkin-Huxley Parameters
CONFIG["HH"]["EL"] = -75.0   # Deep resting potential (간섭 차단)
CONFIG["HH"]["spike_thresh"] = 0.0  # 📐 if V > 0.0 → spike

# ======================================================================
# 2. Neuron Classes
# ======================================================================
class LightNeuron:
    """
    생리학적 뉴런 모델 with Short-Term Plasticity
    
    📐 구현 수식:
    - Soma: Hodgkin-Huxley dynamics (HHSomaQuick)
    - STP/PTP: 단기 시냅스 가소성
    
    Attributes
    ----------
    soma : HHSomaQuick
        Hodgkin-Huxley 기반 soma (전압, 이온 채널 포함)
    S : float
        시냅스 강도 (0.0~1.0), Short-Term Potentiation
    PTP : float
        Post-Tetanic Potentiation (1.0~2.0), 단기 기억 흔적
    """
    def __init__(self, name, gL_mod=0.04):
        cfg = CONFIG["HH"].copy()
        if gL_mod > 0: cfg["gL"] = gL_mod
        self.soma = HHSomaQuick(cfg)  # HH dynamics 사용
        self.S, self.PTP = 0.0, 1.0

    def step(self, dt, I_ext=0.0):
        """
        한 타임스텝 진행
        
        📐 STP/PTP 수식:
        Spike 시:  S ← S + 0.3,    PTP ← PTP + 0.05
        Decay:     S ← S - 0.01,   PTP ← PTP - 0.001
        """
        self.soma.step(dt, I_ext)
        sp = self.soma.spiking()
        if sp:
            # 📐 Spike 발생 시 단기 강화
            self.S = min(1.0, self.S + 0.3)      # S ← S + 0.3
            self.PTP = min(2.0, self.PTP + 0.05)  # PTP ← PTP + 0.05
        else:
            # 📐 Decay (감쇠)
            self.S = max(0.0, self.S - 0.01)       # S ← S - 0.01
            self.PTP = max(1.0, self.PTP - 0.001)  # PTP ← PTP - 0.001
        return sp, self.S, self.PTP

# ======================================================================
# 3. Subiculum & Cortex (Output)
# ======================================================================
class SubiculumFast:
    """
    해마체 (Subiculum) — 단기 메모리 통합기
    
    📐 구현 수식 (1차 Low-pass Filter):
    y(t+dt) = (1-α)·y(t) + spike(t)
    where α = dt/τ
    
    역할: CA1의 스파이크 패턴을 시간적으로 통합하여
          안정적인 메모리 흔적(trace) 생성
    
    Parameters
    ----------
    dt : float
        시뮬레이션 타임스텝 (ms)
    tau : float
        시간 상수 (ms), 기본값 20.0
    """
    def __init__(self, dt, tau=20.0):
        self.dt = dt
        self.alpha = dt/tau  # 📐 α = dt/τ
        self.y = 0.0
    
    def step(self, spike):
        """📐 수식: y(t+dt) = (1-α)·y(t) + spike(t)"""
        self.y = (1.0-self.alpha)*self.y + (1.0 if spike else 0.0)
        return self.y
    
    def reset(self): 
        self.y = 0.0

class CortexRidge:
    """
    대뇌피질 (Cortex) — Ridge Regression 기반 장기 기억 분류기
    
    📐 구현 수식:
    
    1) Initial Training (Wake 후):
       W = Y·X^T·(X·X^T + αI)^(-1)
       
       where:
       - X: input patterns from subiculum
       - Y: target labels (one-hot encoding)
       - α: regularization parameter (높을수록 약한 학습)
       - I: identity matrix
    
    2) Incremental Learning (Sleep 중 Replay):
       error = y - ŷ
       W ← W + η·(error ⊗ input)
       
       where:
       - η: learning rate (낮을수록 느린 학습)
       - ⊗: outer product
    
    3) Inference (Recall 시):
       z = W·x
       p_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
       
       Numerically stable softmax
    
    역할: 해마의 단기 패턴을 장기 기억으로 통합 및 분류
         느린 학습 but 영구 저장
    """
    def __init__(self, input_dim, output_dim):
        self.W = np.zeros((output_dim, input_dim))
        self.inputs = []   # X: 입력 패턴들
        self.targets = []  # Y: 타겟 라벨들
    
    def collect_data(self, sub, label):
        """해마로부터 데이터 수집 (낮 동안 경험)"""
        self.inputs.append(sub)
        self.targets.append(label)
    
    def train(self, alpha=0.1):
        """
        피질 초기 학습 (Wake 후 1회)
        
        📐 수식: W = Y·X^T·(X·X^T + αI)^(-1)
        
        Parameters
        ----------
        alpha : float
            Regularization parameter
            높을수록 약한 학습 (기본값 0.1)
        """
        if len(self.inputs) < 2:
            print("⚠️  Not enough data for training")
            return
        X = np.array(self.inputs).T   # 입력 행렬
        Y = np.array(self.targets).T  # 타겟 행렬
        dim = X.shape[0]
        I = np.eye(dim)
        
        # 📐 Ridge Regression: W = Y·X^T·(X·X^T + αI)^(-1)
        self.W = Y @ X.T @ np.linalg.pinv(X @ X.T + alpha * I)
        print(f"🧠 Cortex: Trained on {len(self.inputs)} patterns")
    
    def incremental_learn(self, sub, label, lr=0.01):
        """
        점진적 학습 (Sleep 중 Replay마다 호출)
        
        📐 수식:
        error = y - ŷ
        W ← W + η·(error ⊗ input)
        
        Parameters
        ----------
        lr : float
            Learning rate (낮을수록 느린 학습, 기본값 0.01)
        """
        pred = (self.W @ np.array(sub).reshape(-1,1)).flatten()
        error = np.array(label) - pred  # 📐 error = y - ŷ
        # 📐 Gradient descent: W ← W + η·(error ⊗ input)
        self.W += lr * np.outer(error, sub)
    
    def infer(self, sub):
        """
        패턴 인식 (Recall 시)
        
        📐 수식:
        z = W·x
        p_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
        
        Returns
        -------
        prob : ndarray
            각 클래스에 대한 확률 분포 (Softmax)
        """
        raw_output = (self.W @ np.array(sub).reshape(-1,1)).flatten()
        
        # 📐 Numerically stable softmax
        exp_output = np.exp(raw_output - np.max(raw_output))
        return exp_output / np.sum(exp_output)

# ======================================================================
# 4. WTA (Winner-Take-All) Helper
# ======================================================================
def apply_wta(neurons, k=3):
    """
    Winner-Take-All: 상위 K개 뉴런만 유지, 나머지 억제
    
    📐 개념:
    1. 전압(V) 기준으로 뉴런 정렬
    2. 상위 k개 선택 (winners)
    3. 나머지는 V = -70mV로 강제 억제 (losers)
    
    생물학적 의미:
    - CA3/CA1의 sparse coding 구현
    - 패턴 간 간섭 최소화
    - 에너지 효율적 표현
    
    📐 수식:
    Select top-k neurons by V
    V_loser ← -70 mV (for all losers)
    """
    # 📐 Step 1: 전압 기준 정렬
    voltages = [(i, n.soma.V) for i, n in enumerate(neurons)]
    voltages.sort(key=lambda x: x[1], reverse=True)
    
    # 📐 Step 2: 하위 뉴런 선택 (losers)
    losers = [idx for idx, _ in voltages[k:]]
    
    # 📐 Step 3: Losers 억제 (V ← -70mV)
    for idx in losers:
        if neurons[idx].soma.V > -60.0:
            neurons[idx].soma.V = -70.0
            neurons[idx].soma.spike_flag = False
            neurons[idx].soma.mode = "rest"

# ======================================================================
# 5. Dream Functions (Sleep - Memory Consolidation)
# ======================================================================
def hippocampal_replay(dg, ca3, ca1, sub, mossy, schaffer, ca3_syns, 
                       pattern_indices, ca3_cluster, cluster_name,
                       dt=0.1, replay_steps=200, noise_level=0.3):
    """
    해마 재생 (Hippocampal Replay) — Sleep Phase
    
    📐 수식:
    DG 입력:
      I = I_base + N(0, σ)
      where I_base = 50.0 (wake의 200.0보다 약함)
            σ = noise_level * 10.0
    
    CA3 Synaptic Reinforcement:
      Q_ij ← Q_ij + f(S, PTP)
      S, PTP를 통해 시냅스 강도 증가
    
    생물학적 근거:
    - 수면 중 해마가 낮의 경험을 재생
    - 약한 자극 + 노이즈 = 자발적 활성화
    - Replay를 통해 시냅스 강화 → 피질로 전달
    
    Parameters
    ----------
    dt : float
        Time step (ms)
    replay_steps : int
        Replay 지속 시간 (기본값 200 steps = 20ms)
    noise_level : float
        Noise 강도 (기본값 0.3)
    
    Returns
    -------
    replay_activity : ndarray
        Subiculum output (피질 학습용)
    """
    N = len(ca3)
    print(f"   🌀 Replaying pattern {pattern_indices} (Cluster {cluster_name})...", end="")
    
    # 📐 약한 자극으로 자발적 재생
    for k in range(replay_steps):
        t = k * dt
        
        # 📐 DG: I = I_base + N(0, σ)
        # Wake: I = 200 (강한 자극)
        # Sleep: I = 50 (약한 자극) + Gaussian noise
        for i in range(N):
            I_base = 50.0 if (i in pattern_indices and t < 5.0) else 0.0
            I_noise = np.random.randn() * noise_level * 10.0  # 📐 N(0, σ)
            sp, S, PTP = dg[i].step(dt, I_base + I_noise)
            if sp: 
                # 📐 Spike 전달: S, PTP 포함
                mossy[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
        
        # 📐 Synaptic Delivery (시냅스 전류 전달)
        for s in mossy + schaffer: s.deliver(t)
        for l in ca3_syns: l['syn'].deliver(t)
        
        # 📐 CA3: Recurrent network activation & reinforcement
        # Q_ij ← Q_ij + f(S, PTP)
        for i in ca3_cluster:
            sp, S, PTP = ca3[i].step(dt, ca3[i].soma.get_total_synaptic_current())
            if sp:
                # 📐 Synaptic Reinforcement (클러스터 내부만)
                # S, PTP 값이 높을수록 강한 시냅스 전달
                for l in ca3_syns:
                    if l['pre'] == i and l['cluster'] == cluster_name:
                        l['syn'].on_pre_spike(t, S, PTP, 100.0, 0.0)
                schaffer[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
        
        # 비활성 CA3는 휴지
        for i in range(N):
            if i not in ca3_cluster:
                ca3[i].step(dt, 0)
        
        # CA1 -> Subiculum (클러스터 영역만)
        for i in ca3_cluster:
            sp, _, _ = ca1[i].step(dt, ca1[i].soma.get_total_synaptic_current())
            sub[i].step(sp)
        
        # 비활성 CA1
        for i in range(N):
            if i not in ca3_cluster:
                ca1[i].step(dt, 0)
    
    # Subiculum 출력 수집
    replay_activity = np.array([s.y for s in sub])
    print(f" Done (活性: {np.sum(replay_activity > 0.1):.0f} neurons)")
    return replay_activity

def synaptic_decay(ca3_syns, decay_rate=0.95):
    """
    시냅스 약화 (Hippocampal Decay) — Sleep 후
    
    📐 수식:
    Q_max ← Q_max · decay_rate
    
    생물학적 근거:
    - 해마는 임시 저장소 (weeks~months)
    - 피질로 전송되지 않으면 소멸
    - Systems Consolidation Theory:
      "해마 → 피질로 메모리 이동, 해마는 약화"
    
    Parameters
    ----------
    decay_rate : float
        감쇠율 (0.95 = 5% 약화, 0.7 = 30% 약화)
    
    Example
    -------
    Q_max = 30.0
    After decay (0.95): Q_max = 28.5
    After decay (0.7): Q_max = 21.0
    """
    for link in ca3_syns:
        # 📐 Q_max ← Q_max · decay_rate
        link['syn'].Q_max *= decay_rate

def cortex_consolidation(cortex, replay_activities, pattern_labels, lr=0.05):
    """
    피질 공고화 (Cortical Consolidation) — Sleep 중
    
    📐 수식:
    error = y - ŷ
    W ← W + η·(error ⊗ input)
    
    생물학적 근거:
    - 수면 중 해마 replay → 피질 점진 학습
    - 느린 학습 (small lr) → 안정적 장기 저장
    - 여러 replay 반복 → 강건한 표현 형성
    
    Parameters
    ----------
    lr : float
        Learning rate (기본값 0.05)
        Wake initial training: alpha=0.5 (약한 학습)
        Sleep consolidation: lr=0.05 (느린 but 반복적)
    
    Process:
    1. Replay activity → 피질 입력
    2. Gradient descent update
    3. 여러 cycle 반복
    4. 해마 약화 → 피질 강화
    """
    print(f"   🧠 Cortex: Consolidating {len(replay_activities)} replays...")
    for activity, label in zip(replay_activities, pattern_labels):
        # 📐 Incremental learning: W ← W + η·(error ⊗ input)
        cortex.incremental_learn(activity, label, lr=lr)
    print(f"   ✅ Consolidation Complete.")

# ======================================================================
# 6. Main Simulation (Wake → Sleep → Wake)
# ======================================================================
def run_dream_simulation(N=20, dt=0.1):
    """
    해마-피질 메모리 공고화 시뮬레이션
    
    📐 3-Phase Pipeline:
    
    Phase 1 - Wake (Learning):
      DG → CA3 → CA1 → Subiculum → Cortex
      - 강한 입력 (I = 200 pA)
      - CA3 recurrent learning
      - Cortex 약한 학습 (alpha=0.5)
    
    Phase 2 - Sleep (Consolidation):
      1. Hippocampal Replay (해마 재생)
         📐 I = 50 + N(0, σ)
      2. Cortical Consolidation (피질 강화)
         📐 W ← W + η·(error ⊗ input)
      3. Hippocampal Decay (해마 약화)
         📐 Q_max ← Q_max · decay_rate
    
    Phase 3 - Wake (Recall):
      - Hippocampus test: 약화되었지만 작동
      - Cortex direct test: 단서만으로 회상
    
    생물학적 의미:
    - Systems Consolidation Theory 구현
    - 해마 → 피질 메모리 전이
    - 수면의 역할 시뮬레이션
    
    Parameters
    ----------
    N : int
        총 뉴런 개수 (기본값 20)
    dt : float
        Time step in ms (기본값 0.1)
    """
    random.seed(42); np.random.seed(42)
    print(f"\n🌙 HIPPOCAMPUS DREAM SIMULATION 🌙")
    print("=" * 70)
    print("Simulating: Wake (Learning) → Sleep (Dream/Consolidation) → Wake (Recall)")
    print("=" * 70)

    # ===== 모듈 생성 (Network Architecture) =====
    dg = [LightNeuron(f"DG{i}", 0.1) for i in range(N)]    # Dentate Gyrus
    ca3 = [LightNeuron(f"CA3{i}") for i in range(N)]        # CA3 (recurrent)
    ca1 = [LightNeuron(f"CA1{i}", 0.08) for i in range(N)]  # CA1 (relay)
    sub = [SubiculumFast(dt) for i in range(N)]             # Subiculum (integrator)
    cortex = CortexRidge(N, 3)                              # Cortex (classifier)

    # ======================================================================
    # 패턴 정의 (병렬 클러스터 구조)
    # ======================================================================
    # 📐 개념: Sparse & Non-overlapping clusters
    # - 각 패턴은 3개 뉴런으로 표현 (sparse coding)
    # - 클러스터 간 물리적 격리 (간섭 방지)
    
    patterns = {
        "A": ([0, 1, 2], [1,0,0]),      # DG[0,1,2] → CA3 Cluster A
        "B": ([6, 7, 8], [0,1,0]),      # DG[6,7,8] → CA3 Cluster B
        "C": ([12, 13, 14], [0,0,1])    # DG[12,13,14] → CA3 Cluster C
    }
    
    # CA3 클러스터 정의 (물리적 분리)
    ca3_clusters = {
        "A": [0, 1, 2],      # CA3 뉴런 0,1,2
        "B": [6, 7, 8],      # CA3 뉴런 6,7,8
        "C": [12, 13, 14]    # CA3 뉴런 12,13,14
    }
    
    # 📐 DG → CA3 매핑 (Mossy Fiber connections)
    # DG[i] → CA3[dg_to_ca3_map[i]]
    dg_to_ca3_map = {}
    for pattern_name, ca3_indices in ca3_clusters.items():
        pattern_indices = patterns[pattern_name][0]
        for dg_idx, ca3_idx in zip(pattern_indices, ca3_indices):
            dg_to_ca3_map[dg_idx] = ca3_idx

    # --- 연결 구축 (병렬 클러스터) ---
    # Mossy Fibers: DG -> CA3 (클러스터별 매핑)
    mossy = []
    for i in range(N):
        if i in dg_to_ca3_map:
            ca3_target = dg_to_ca3_map[i]
            syn = SynapseCore(dg[i].soma, ca3[ca3_target].soma, Q_max=80.0)
        else:
            # 패턴에 속하지 않는 DG 뉴런은 자기 자신에게 연결 (사용 안 됨)
            syn = SynapseCore(dg[i].soma, ca3[i].soma, Q_max=0.1)
        mossy.append(syn)
    
    # Schaffer Collaterals: CA3 -> CA1 (1:1)
    schaffer = [SynapseCore(ca3[i].soma, ca1[i].soma, delay_ms=2.0, Q_max=25.0) for i in range(N)]
    
    # CA3 Recurrent: 클러스터 내부만 연결 (물리적 격리)
    ca3_syns = []
    for cluster_name, cluster_indices in ca3_clusters.items():
        for i in cluster_indices:
            for j in cluster_indices:
                if i == j: continue
                # 클러스터 내부만 강하게 연결
                syn = SynapseCore(ca3[i].soma, ca3[j].soma, delay_ms=1.5, Q_max=30.0)
                ca3_syns.append({'pre': i, 'post': j, 'syn': syn, 'cluster': cluster_name})

    print(f"System Ready: {len(ca3_syns)} Selective Connections.")

    # =========================================================
    # PHASE 1: WAKE - LEARNING (낮: 경험)
    # =========================================================
    # 📐 목표: 각 패턴을 CA3 클러스터에 인코딩하고 Cortex 약한 학습
    # 
    # 과정:
    # 1. DG에 패턴 입력 (I = 200 pA, t < 10ms)
    # 2. CA3 recurrent activation (pattern storage)
    # 3. CA1 → Subiculum integration
    # 4. Cortex에 데이터 수집 (약한 학습)
    # 5. Reset 후 다음 패턴
    
    print("\n" + "="*70)
    print("☀️  PHASE 1: WAKE - LEARNING (Day)")
    print("="*70)
    
    steps = int(40.0/dt)  # 40ms 시뮬레이션
    
    # 📐 Global inhibition (population spike rate 제어)
    # I_inhib = -spike_count * INHIB_CONSTANT
    DG_INHIB = 80.0   # DG 억제 강도
    CA3_INHIB = 20.0  # CA3 억제 강도
    dg_last = 0       # 이전 타임스텝 스파이크 수
    ca3_last = 0

    for name, (p, label) in patterns.items():
        print(f"  📝 Encoding '{name}': {p}...", end="")
        for s in sub: s.reset()
        
        # 현재 패턴의 CA3 클러스터
        active_cluster = ca3_clusters[name]
        
        for k in range(steps):
            t = k*dt
            
            # DG (패턴 입력)
            dg_now=0; I_dg=-dg_last*DG_INHIB
            for i in range(N):
                I = 200 if (i in p and t<10) else 0
                sp, S, PTP = dg[i].step(dt, I+I_dg)
                if sp: dg_now+=1; mossy[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
            dg_last = dg_now

            # Deliver
            for s in mossy+schaffer: s.deliver(t)
            for l in ca3_syns: l['syn'].deliver(t)

            # CA3 (클러스터별 업데이트)
            ca3_now=0; I_ca3=-ca3_last*CA3_INHIB
            for i in active_cluster:  # 활성 클러스터만 업데이트
                sp, S, PTP = ca3[i].step(dt, ca3[i].soma.get_total_synaptic_current()+I_ca3)
                if sp:
                    ca3_now+=1
                    # 현재 클러스터 내 시냅스만 강화
                    for l in ca3_syns:
                        if l['pre'] == i and l['cluster'] == name: 
                            l['syn'].on_pre_spike(t, S, PTP, 100.0, 0.0)
                    schaffer[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
            
            # 비활성 클러스터는 휴지 상태 유지
            for i in range(N):
                if i not in active_cluster:
                    ca3[i].step(dt, I_ca3)
            
            ca3_last = ca3_now

            # CA1 -> Subiculum (활성 클러스터에 해당하는 CA1만)
            for i in active_cluster:
                sp,_,_ = ca1[i].step(dt, ca1[i].soma.get_total_synaptic_current())
                sub[i].step(sp)
            
            # 비활성 CA1은 휴지
            for i in range(N):
                if i not in active_cluster:
                    ca1[i].step(dt, 0)
        
        # Data Collect (해마가 일시적으로 저장)
        sub_activity = np.array([s.y for s in sub])
        cortex.collect_data(sub_activity, np.array(label))
        print(f" ✅ Stored in Hippocampus")
        
        # Deep Wash
        for _ in range(200):
             for n in dg+ca3+ca1: n.step(dt,0)
             for s in mossy+schaffer: s.deliver(0)
             for l in ca3_syns: l['syn'].deliver(0)
        
        # Force Reset
        for n in dg+ca3+ca1: 
            n.soma.V=-75.0
            n.soma.m, n.soma.h, n.soma.n = 0.05, 0.6, 0.32
            n.soma.spike_flag=False
            n.soma.I_syn_total=0
            n.soma.mode = "rest"
            n.soma.active_remaining = 0
        for s in mossy+schaffer: s.spikes=[]; s.I_syn=0
        for l in ca3_syns: l['syn'].spikes=[]; l['syn'].I_syn=0

    # 피질 초기 학습 (약한 학습)
    cortex.train(alpha=0.5)  # 높은 alpha = 약한 학습
    print("\n💤 End of Day. Going to sleep...")

    # =========================================================
    # PHASE 2: SLEEP - DREAMING (밤: 꿈 - 메모리 공고화)
    # =========================================================
    # 📐 목표: 해마 replay → 피질 consolidation → 해마 decay
    # 
    # 3-Step Process:
    # 1. Hippocampal Replay:
    #    📐 I = I_base(50) + N(0, σ)
    #    - 약한 자극으로 패턴 재생
    #    - 시냅스 reinforcement
    # 
    # 2. Cortical Consolidation:
    #    📐 W ← W + η·(error ⊗ input)
    #    - Replay activity를 피질이 점진 학습
    #    - 여러 cycle 반복
    # 
    # 3. Hippocampal Decay:
    #    📐 Q_max ← Q_max · decay_rate
    #    - 해마 시냅스 약화
    #    - 피질로 전송되지 않은 정보 소멸
    
    print("\n" + "="*70)
    print("🌙 PHASE 2: SLEEP - DREAMING (Memory Consolidation)")
    print("="*70)
    
    replay_activities = []
    pattern_labels = []
    
    # 각 패턴을 여러 번 재생 (꿈)
    num_replays = 3
    for replay_idx in range(num_replays):
        print(f"\n💭 Dream Cycle {replay_idx + 1}/{num_replays}:")
        
        for name, (p, label) in patterns.items():
            # Reset subiculum
            for s in sub: s.reset()
            
            # 현재 패턴의 CA3 클러스터
            cluster = ca3_clusters[name]
            
            # Hippocampal Replay (해마 재생)
            replay_activity = hippocampal_replay(
                dg, ca3, ca1, sub, mossy, schaffer, ca3_syns,
                pattern_indices=p, ca3_cluster=cluster, cluster_name=name,
                dt=dt, replay_steps=150, noise_level=0.3
            )
            
            replay_activities.append(replay_activity)
            pattern_labels.append(label)
            
            # Reset after replay
            for n in dg+ca3+ca1:
                n.soma.V = -75.0
                n.soma.spike_flag = False
                n.soma.I_syn_total = 0
            for s in mossy+schaffer: s.spikes=[]; s.I_syn=0
            for l in ca3_syns: l['syn'].spikes=[]; l['syn'].I_syn=0
    
    # Cortex Consolidation (피질 공고화)
    print(f"\n🔄 Cortex Consolidation:")
    cortex_consolidation(cortex, replay_activities, pattern_labels, lr=0.03)
    
    # Hippocampal Decay (해마 약화)
    print(f"\n🔻 Hippocampal Synaptic Decay:")
    initial_Q = ca3_syns[0]['syn'].Q_max
    synaptic_decay(ca3_syns, decay_rate=0.7)  # 30% 약화
    final_Q = ca3_syns[0]['syn'].Q_max
    print(f"   CA3 Synapse: {initial_Q:.1f} → {final_Q:.1f} (70% retention)")
    
    print("\n☀️  Morning! Waking up...")

    # =========================================================
    # PHASE 3: WAKE - RECALL (낮: 회상)
    # =========================================================
    # 📐 목표: 수면 후 메모리 테스트
    # 
    # 두 가지 테스트:
    # 1. Hippocampus → Cortex:
    #    - 해마가 패턴 복원 → 피질이 인식
    #    - 해마는 약화되었지만 아직 작동
    #    - 단기 기억 테스트
    # 
    # 2. Direct Cortex Recall:
    #    - 단서만으로 피질이 직접 회상
    #    - 해마 우회 (hippocampal bypass)
    #    - 장기 기억 테스트
    # 
    # 생물학적 의미:
    # - Cortex score 높음 → Consolidation 성공
    # - Hippo만 작동 → 더 많은 수면 필요
    # - 둘 다 실패 → Memory system degraded
    
    print("\n" + "="*70)
    print("☀️  PHASE 3: WAKE - RECALL (After Sleep)")
    print("="*70)
    
    # 📐 Recall 시 더 강한 억제 (sparse activation)
    DG_INHIB_R = 150.0   # DG 억제
    CA3_INHIB_R = 60.0   # CA3 억제
    CA1_INHIB_R = 35.0   # CA1 억제
    score_hippo = 0   # 해마 경로 점수
    score_cortex = 0  # 피질 직접 회상 점수

    for name, (p, label) in patterns.items():
        cue = [p[0]]
        print(f"\n🧪 Test: Cue {cue} → Expecting '{name}'")
        
        # 현재 패턴의 CA3 클러스터
        active_cluster = ca3_clusters[name]
        
        # Reset
        for n in dg+ca3+ca1:
            n.soma.V=-70.0
            n.soma.m, n.soma.h, n.soma.n = 0.05, 0.6, 0.32
            n.soma.spike_flag=False
            n.soma.I_syn_total=0
            n.soma.mode = "rest"
        for s in sub: s.reset()
        dg_last=0; ca3_last=0

        for k in range(steps):
            t = k*dt
            
            # DG (단서 입력)
            dg_now=0; I_dg=-dg_last*DG_INHIB_R
            for i in range(N):
                I = 200 if (i in cue and t<10) else 0
                sp,S,PTP = dg[i].step(dt, I+I_dg)
                if sp: dg_now+=1; mossy[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
            dg_last = dg_now

            # Deliver
            for s in mossy+schaffer: s.deliver(t)
            for l in ca3_syns: l['syn'].deliver(t)

            # CA3 (클러스터별, weakened)
            ca3_now=0; I_ca3=-ca3_last*CA3_INHIB_R
            for i in active_cluster:
                sp,S,PTP = ca3[i].step(dt, ca3[i].soma.get_total_synaptic_current()+I_ca3)
                if sp:
                    ca3_now+=1
                    # 현재 클러스터 내 시냅스만 활성화
                    for l in ca3_syns:
                        if l['pre'] == i and l['cluster'] == name:
                            l['syn'].on_pre_spike(t, S, PTP, 100.0, 0.0)
                    schaffer[i].on_pre_spike(t, S, PTP, 100.0, 0.0)
            
            # 비활성 클러스터는 억제
            for i in range(N):
                if i not in active_cluster:
                    ca3[i].step(dt, I_ca3)
            
            ca3_last = ca3_now
            
            # CA3 WTA (클러스터 내부만)
            if t > 2.0:
                cluster_neurons = [ca3[i] for i in active_cluster]
                if len(cluster_neurons) > 3:
                    apply_wta(cluster_neurons, k=3)

            # CA1 (클러스터 해당 영역만)
            I_ca1 = -CA1_INHIB_R
            for i in active_cluster:
                sp,_,_ = ca1[i].step(dt, ca1[i].soma.get_total_synaptic_current()+I_ca1)
                sub[i].step(sp)
            
            # 비활성 CA1
            for i in range(N):
                if i not in active_cluster:
                    ca1[i].step(dt, 0)
            
            # CA1 WTA (클러스터 내부만)
            if t > 3.0:
                cluster_ca1 = [ca1[i] for i in active_cluster]
                if len(cluster_ca1) > 3:
                    apply_wta(cluster_ca1, k=3)

        # Hippocampus Output (약화됨)
        readout_hippo = np.array([s.y for s in sub])
        pred_vec_hippo = cortex.infer(readout_hippo)
        pred_idx_hippo = np.argmax(pred_vec_hippo)
        pred_name_hippo = ["A", "B", "C"][pred_idx_hippo]
        conf_hippo = pred_vec_hippo[pred_idx_hippo]

        print(f"   🏛️  Hippocampus → Cortex: \"{pred_name_hippo}\" (Conf: {conf_hippo:.2f})")
        
        if pred_name_hippo == name:
            print(f"      ✅ Correct (Hippocampus still functional)")
            score_hippo += 1
        else:
            print(f"      ⚠️  Weakened (Hippocampus decayed)")

        # 📐 Direct Cortex Recall (해마 우회 - 순수 피질 회상)
        # 
        # 생물학적 의미:
        # - Remote memory recall after hippocampal damage
        # - 충분한 consolidation 후 피질만으로 회상 가능
        # - Systems Consolidation Theory의 핵심 예측
        # 
        # 테스트 방법:
        # - 단서 1개만 제공 (cue neuron)
        # - 해마 없이 피질만 작동
        # - 피질이 consolidation으로 학습한 표현만 사용
        
        print(f"   🧠 Testing Direct Cortical Recall...")
        
        # 📐 Minimal cue pattern (sparse input)
        mini_pattern = np.zeros(N)
        mini_pattern[cue[0]] = 1.0  # 단서 1개만 활성화
        
        # 📐 Cortex inference: p = softmax(W·x)
        pred_vec_cortex = cortex.infer(mini_pattern)
        pred_idx_cortex = np.argmax(pred_vec_cortex)
        pred_name_cortex = ["A", "B", "C"][pred_idx_cortex]
        conf_cortex = pred_vec_cortex[pred_idx_cortex]
        
        print(f"   🧠 Cortex Direct: \"{pred_name_cortex}\" (Conf: {conf_cortex:.2f})")
        
        if pred_name_cortex == name:
            print(f"      ✅ Correct (Consolidated to Cortex!)")
            score_cortex += 1
        else:
            print(f"      ❌ Failed (Needs more consolidation)")

    print("\n" + "="*70)
    print(f"🏆 FINAL RESULTS:")
    print(f"   Hippocampus → Cortex: {score_hippo}/3")
    print(f"   Direct Cortex Recall: {score_cortex}/3")
    print("="*70)
    
    if score_cortex == 3:
        print("\n🎉 Perfect Consolidation! Memories transferred to long-term storage!")
        print("   (해마 없이도 피질만으로 회상 가능 = 장기 기억 완성)")
    elif score_hippo == 3:
        print("\n✅ Hippocampus functional, but consolidation incomplete.")
        print("   (해마는 작동하지만 피질 전송 미완료 = 더 많은 수면 필요)")
    else:
        print("\n⚠️  Memory system degraded. Need more sleep/consolidation cycles.")

if __name__ == "__main__":
    run_dream_simulation()
