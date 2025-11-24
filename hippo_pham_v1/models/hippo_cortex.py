"""
===================================================================================
HIPPO_CORTEX — Multi-Neuron Hodgkin-Huxley Chain Simulation
===================================================================================
지은이: GNJz | 발행: 2025.11.24

순수 Hodgkin-Huxley 뉴런 체인 시뮬레이션 (교육용)

📐 구현된 핵심 수식:

1️⃣ Hodgkin-Huxley Membrane Dynamics:
   C_m · dV/dt = -I_Na - I_K - I_L - I_syn + I_ext
   
   where:
   I_Na = g_Na · m³ · h · (V - E_Na)    [Sodium current]
   I_K  = g_K · n⁴ · (V - E_K)           [Potassium current]
   I_L  = g_L · (V - E_L)                 [Leak current]

2️⃣ Gating Variable Dynamics:
   dm/dt = α_m(V)·(1-m) - β_m(V)·m
   dh/dt = α_h(V)·(1-h) - β_h(V)·h
   dn/dt = α_n(V)·(1-n) - β_n(V)·n
   
   where α, β are voltage-dependent rate functions

3️⃣ Rate Functions (Hodgkin-Huxley Original):
   α_m(V) = 0.1·(V+40) / (1 - exp(-(V+40)/10))
   β_m(V) = 4.0·exp(-(V+65)/18)
   
   α_h(V) = 0.07·exp(-(V+65)/20)
   β_h(V) = 1 / (1 + exp(-(V+35)/10))
   
   α_n(V) = 0.01·(V+55) / (1 - exp(-(V+55)/10))
   β_n(V) = 0.125·exp(-(V+65)/80)

4️⃣ Synaptic Transmission:
   I_syn = g_syn · s · (V - E_syn)
   
   ds/dt = -s / τ_syn
   
   On presynaptic spike: s ← min(s_max, s + s_rise)

5️⃣ Spike Detection:
   if V(t) > V_threshold → spike = True

6️⃣ Chain Architecture:
   N0 → N1 → N2 → ... → N(N-1)
   
   단방향 체인: i번 뉴런 spike → (i+1)번 뉴런 시냅스 활성화

7️⃣ External Input (Pulse):
   I_ext(t) = { I_base + I_pulse  if t_on ≤ t ≤ t_off
              { I_base             otherwise

8️⃣ Numerical Integration:
   Euler method: y(t+dt) = y(t) + dt·f(y,t)

===================================================================================
"""

"""
===================================================================================
📦 Note on Implementation
===================================================================================

This is a **standalone educational implementation** of the Hodgkin-Huxley model.
Unlike other files in this suite, it does NOT depend on `v3_event.py`.

Purpose:
  - Demonstrate basic HH dynamics from scratch
  - Show propagation in a chain of neurons
  - Serve as a minimal reference implementation

For production hippocampal simulations, use the v3_event-based implementations
(hippo_ca1.py, hippo_sub.py, etc.) which are optimized and feature-complete.

===================================================================================
"""

# Qquarts co Present
# 지은이 : GNJz 
# 발행 2025.11.24

import math
from dataclasses import dataclass, field
from typing import Dict, List

# ============================================================
# 🔧 전역 설정 (하드코딩 금지 → CONFIG 한 군데에서만 수치 관리)
# ============================================================

CONFIG: Dict[str, float] = {
    # 📐 네트워크 크기
    "N_NEURONS": 8,          # ← 20, 102로 올리면 바로 확장

    # 📐 시뮬레이션 시간 설정
    "T_TOTAL_MS": 200.0,     # 총 시뮬레이션 시간 [ms]
    "DT_MS": 0.05,           # 타임스텝 [ms]

    # 📐 막 전기용량
    "C_M": 1.0,              # [µF/cm²] (상대값)

    # 📐 이온 평형전위 [mV]
    "E_NA": 50.0,            # Sodium reversal potential
    "E_K": -77.0,            # Potassium reversal potential
    "E_L": -54.4,            # Leak reversal potential

    # 📐 최대 전도도 [mS/cm²]
    "G_NA": 120.0,           # Sodium conductance (HH standard)
    "G_K": 36.0,             # Potassium conductance (HH standard)
    "G_L": 0.3,              # Leak conductance

    # 📐 시냅스 관련
    "G_SYN": 0.3,            # 시냅스 최대 전도도 (상대값)
    "E_SYN": 0.0,            # 흥분성 시냅스 (0 mV)
    "TAU_SYN_MS": 5.0,       # 시냅스 게이트 붕괴 τ [ms]
    "S_RISE": 0.5,           # threshold 넘으면 s 상승량
    "S_MAX": 1.0,            # s 상한

    # 📐 발화 판정
    "SPIKE_THRESHOLD_MV": 0.0,  # Vm이 이 값을 넘으면 spike로 판정

    # 📐 외부 입력 (첫 번째 뉴런만)
    "I_EXT_BASE": 0.0,       # 기본 외부 입력 [µA/cm²]
    "I_EXT_PULSE": 10.0,     # 펄스 크기
    "I_EXT_ON_MS": 20.0,     # 펄스 시작 시간
    "I_EXT_OFF_MS": 40.0,    # 펄스 종료 시간

    # 📐 초기 막전위
    "V_REST": -65.0,

    # 📐 안전용 클리핑 (수치 폭주 방지용)
    "V_MIN": -120.0,
    "V_MAX": 60.0,

    # 📐 간단한 에너지 관점 지표용 스케일
    "ENERGY_SCALE": 1e-3,    # Σ |I_ion * (V-E)| dt
}


# ============================================================
# ⚙ HH 게이트 함수 (근사형)
# ============================================================

def alpha_m(V: float) -> float:
    """
    m-gate activation rate (Sodium activation)
    
    📐 수식:
    α_m(V) = 0.1·(V+40) / (1 - exp(-(V+40)/10))
    
    생물학적 의미:
    - Sodium channel activation의 전압 의존 속도
    - V가 증가하면 α_m 증가 → m 증가 → Na+ 채널 열림
    """
    x = V + 40.0
    # 📐 수치 안정성: x ≈ 0일 때 L'Hôpital 규칙 적용
    if abs(x) < 1e-6:
        return 1.0
    return 0.1 * x / (1.0 - math.exp(-x / 10.0))

def beta_m(V: float) -> float:
    """
    m-gate inactivation rate
    
    📐 수식:
    β_m(V) = 4.0 · exp(-(V+65)/18)
    """
    return 4.0 * math.exp(-(V + 65.0) / 18.0)

def alpha_h(V: float) -> float:
    """
    h-gate activation rate (Sodium inactivation)
    
    📐 수식:
    α_h(V) = 0.07 · exp(-(V+65)/20)
    
    생물학적 의미:
    - Sodium channel inactivation의 전압 의존 속도
    - V가 높으면 α_h 감소 → h 감소 → Na+ 채널 불활성화
    """
    return 0.07 * math.exp(-(V + 65.0) / 20.0)

def beta_h(V: float) -> float:
    """
    h-gate inactivation rate
    
    📐 수식:
    β_h(V) = 1 / (1 + exp(-(V+35)/10))
    """
    return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))

def alpha_n(V: float) -> float:
    """
    n-gate activation rate (Potassium activation)
    
    📐 수식:
    α_n(V) = 0.01·(V+55) / (1 - exp(-(V+55)/10))
    
    생물학적 의미:
    - Potassium channel activation의 전압 의존 속도
    - V가 증가하면 α_n 증가 → n 증가 → K+ 채널 열림
    """
    x = V + 55.0
    # 📐 수치 안정성: x ≈ 0일 때 L'Hôpital 규칙 적용
    if abs(x) < 1e-6:
        return 0.1
    return 0.01 * x / (1.0 - math.exp(-x / 10.0))

def beta_n(V: float) -> float:
    """
    n-gate inactivation rate
    
    📐 수식:
    β_n(V) = 0.125 · exp(-(V+65)/80)
    """
    return 0.125 * math.exp(-(V + 65.0) / 80.0)


# ============================================================
# 🧱 단일 뉴런 클래스 (원본 HH 근사)
# ============================================================

@dataclass
class HHNeuron:
    """
    HHNeuron: 단일 구획 Hodgkin-Huxley 뉴런
    
    📐 구현 수식:
    
    1) 막전위 동역학:
       C_m · dV/dt = -g_Na·m³·h·(V-E_Na) - g_K·n⁴·(V-E_K) 
                     - g_L·(V-E_L) - g_syn·s·(V-E_syn) + I_ext
    
    2) 게이팅 변수:
       dm/dt = α_m(V)·(1-m) - β_m(V)·m
       dh/dt = α_h(V)·(1-h) - β_h(V)·h
       dn/dt = α_n(V)·(1-n) - β_n(V)·n
    
    3) 시냅스 게이트:
       ds/dt = -s / τ_syn
       
       On presynaptic spike: s ← min(s_max, s + s_rise)
    
    상태 변수:
      - V : 막전위 [mV]
      - m, h, n : Na/K 게이트
      - s_syn_in : 시냅스 입력 게이트 (0~1)
      - spike: 현재 스텝에서 발화 여부 (bool)
    
    생물학적 의미:
    - m: Sodium activation (빠른 activation)
    - h: Sodium inactivation (중간 inactivation)
    - n: Potassium activation (느린 activation)
    - s: Synaptic conductance (화학적 시냅스 모델)
    """
    
    # 📐 상태 변수
    V: float = CONFIG["V_REST"]      # 막전위 [mV]
    m: float = 0.05                   # Na activation gate
    h: float = 0.6                    # Na inactivation gate
    n: float = 0.32                   # K activation gate
    s_syn_in: float = 0.0             # 시냅스 입력 게이트
    spike: bool = False               # 스파이크 플래그

    # 📐 에너지/통계용
    energy_accum: float = 0.0         # 누적 에너지 (임의 단위)
    spike_count: int = 0              # 스파이크 카운트

    def step(self, dt_ms: float, I_ext: float) -> None:
        """
        단일 스텝 업데이트 (Euler method)
        
        📐 수치 적분:
        y(t+dt) = y(t) + dt · dy/dt
        
        Parameters:
            dt_ms: Timestep [ms]
            I_ext: External current [µA/cm²]
        """
        C_m = CONFIG["C_M"]
        gNa = CONFIG["G_NA"]
        gK = CONFIG["G_K"]
        gL = CONFIG["G_L"]
        ENa = CONFIG["E_NA"]
        EK = CONFIG["E_K"]
        EL = CONFIG["E_L"]
        gSyn = CONFIG["G_SYN"]
        ESyn = CONFIG["E_SYN"]

        V = self.V

        # 📐 Gate kinetics (α-β formulation)
        am = alpha_m(V)
        bm = beta_m(V)
        ah = alpha_h(V)
        bh = beta_h(V)
        an = alpha_n(V)
        bn = beta_n(V)

        # 📐 수식: dm/dt = α_m·(1-m) - β_m·m
        dm = am * (1.0 - self.m) - bm * self.m
        dh = ah * (1.0 - self.h) - bh * self.h
        dn = an * (1.0 - self.n) - bn * self.n

        # 📐 수식: m(t+dt) = m(t) + dt·dm/dt
        self.m += dt_ms * dm
        self.h += dt_ms * dh
        self.n += dt_ms * dn

        # 📐 Ionic Currents
        # I_Na = g_Na · m³ · h · (V - E_Na)
        INa = gNa * (self.m ** 3) * self.h * (V - ENa)
        
        # I_K = g_K · n⁴ · (V - E_K)
        IK  = gK  * (self.n ** 4) * (V - EK)
        
        # I_L = g_L · (V - E_L)
        IL  = gL * (V - EL)
        
        # I_syn = g_syn · s · (V - E_syn)
        Isyn = gSyn * self.s_syn_in * (V - ESyn)

        # 📐 막전위 미분: C_m·dV/dt = -I_Na - I_K - I_L - I_syn + I_ext
        dV = (-INa - IK - IL - Isyn + I_ext) / C_m
        
        # 📐 수식: V(t+dt) = V(t) + dt·dV/dt
        self.V += dt_ms * dV

        # 📐 안전 클리핑 (수치 폭주 방지)
        self.V = max(CONFIG["V_MIN"], min(CONFIG["V_MAX"], self.V))

        # 📐 시냅스 게이트 붕괴: ds/dt = -s/τ
        tau_syn = CONFIG["TAU_SYN_MS"]
        self.s_syn_in += dt_ms * (-self.s_syn_in / tau_syn)
        if self.s_syn_in < 0.0:
            self.s_syn_in = 0.0

        # 📐 발화 판정: if V > V_threshold → spike
        threshold = CONFIG["SPIKE_THRESHOLD_MV"]
        self.spike = (self.V > threshold)

        if self.spike:
            self.spike_count += 1

        # 📐 간단 에너지 지표 누적 (절대값 기반)
        dE = (
            abs(INa * (V - ENa))
            + abs(IK * (V - EK))
            + abs(IL * (V - EL))
        ) * dt_ms * CONFIG["ENERGY_SCALE"]
        self.energy_accum += dE


# ============================================================
# 🔗 N개 뉴런 체인 네트워크
# ============================================================

@dataclass
class NeuronChain:
    """
    NeuronChain: N개의 HHNeuron으로 구성된 단순 체인
    
    📐 구조:
    N0 → N1 → N2 → ... → N(N-1)
    
    📐 시냅스 전달 규칙:
    if neuron[i].spike → neuron[i+1].s_syn_in ← min(s_max, s + s_rise)
    
    📐 외부 입력:
    I_ext(t) = { I_base + I_pulse  if t_on ≤ t ≤ t_off (N0만)
               { I_base             otherwise
    
    생물학적 의미:
    - 피질 레이어 간 정보 전파 모델
    - 단방향 feed-forward network
    - Spike → 화학적 시냅스 → 다음 뉴런 활성화
    """
    
    N: int = CONFIG["N_NEURONS"]
    neurons: List[HHNeuron] = field(default_factory=list)

    def __post_init__(self):
        if not self.neurons:
            self.neurons = [HHNeuron() for _ in range(self.N)]

    def external_current(self, t_ms: float) -> float:
        """
        시간에 따른 외부 입력 I_ext(t)
        
        📐 수식:
        I_ext(t) = { I_base + I_pulse  if t_on ≤ t ≤ t_off
                   { I_base             otherwise
        
        생물학적 의미:
        - 감각 자극의 pulse 입력
        - 첫 번째 뉴런만 외부 자극을 받음
        
        Parameters:
            t_ms: 현재 시간 [ms]
        
        Returns:
            I_ext: 외부 전류 [µA/cm²]
        """
        I_base = CONFIG["I_EXT_BASE"]
        I_pulse = CONFIG["I_EXT_PULSE"]
        t_on = CONFIG["I_EXT_ON_MS"]
        t_off = CONFIG["I_EXT_OFF_MS"]
        
        # 📐 수식: Pulse window
        if t_on <= t_ms <= t_off:
            return I_base + I_pulse
        return I_base

    def propagate_synapses(self):
        """
        이전 스텝에서 spike한 뉴런이 다음 뉴런의 시냅스를 올려줌
        
        📐 수식:
        if neuron[i].spike:
            s[i+1] ← min(s_max, s[i+1] + s_rise)
        
        구조: [i] → [i+1] 단방향 체인
        
        생물학적 의미:
        - Presynaptic spike → Neurotransmitter release
        - Postsynaptic conductance 증가
        - Chemical synaptic transmission
        """
        s_rise = CONFIG["S_RISE"]
        s_max = CONFIG["S_MAX"]

        # 📐 뒤에서부터 업데이트하면 같은 스텝에 중복 적용 방지
        for i in range(self.N - 1 - 1, -1, -1):
            if self.neurons[i].spike:
                post = self.neurons[i + 1]
                # 📐 수식: s ← min(s_max, s + s_rise)
                post.s_syn_in = min(s_max, post.s_syn_in + s_rise)

    def step(self, t_ms: float, dt_ms: float) -> None:
        """
        네트워크 전체 한 스텝 업데이트
        
        📐 실행 순서:
        1) 외부 입력 → 0번 뉴런
        2) 모든 뉴런 HH 업데이트
        3) spike 기반 시냅스 전달
        
        Parameters:
            t_ms: 현재 시간 [ms]
            dt_ms: Timestep [ms]
        """
        # 📐 1) 0번 뉴런에만 외부 입력
        I_ext_0 = self.external_current(t_ms)

        # 📐 2) 뉴런 업데이트
        for idx, neuron in enumerate(self.neurons):
            if idx == 0:
                neuron.step(dt_ms=dt_ms, I_ext=I_ext_0)
            else:
                neuron.step(dt_ms=dt_ms, I_ext=0.0)

        # 📐 3) spike → 시냅스 전달
        self.propagate_synapses()

    def summary(self) -> Dict[str, float]:
        """
        네트워크 전체 통계 요약
        
        Returns:
            dict: {
                'total_spikes': 총 스파이크 수,
                'spikes_per_neuron': 뉴런별 스파이크 리스트,
                'energy_per_neuron': 뉴런별 에너지 리스트
            }
        """
        spikes = [n.spike_count for n in self.neurons]
        energies = [n.energy_accum for n in self.neurons]
        return {
            "total_spikes": sum(spikes),
            "spikes_per_neuron": spikes,
            "energy_per_neuron": energies,
        }


# ============================================================
# 🚀 메인 실행 루프
# ============================================================

def run_simulation():
    """
    HH 뉴런 체인 시뮬레이션 실행
    
    📐 시뮬레이션 구조:
    1) N개의 HH 뉴런을 체인으로 연결
    2) 0번 뉴런에 pulse 입력 (t_on ~ t_off)
    3) Spike propagation 관찰
    4) 결과 출력 (전압, 스파이크, 에너지)
    
    생물학적 의미:
    - 피질 레이어 간 신호 전파
    - Feedforward excitation
    - Spike timing 전파
    """
    T = CONFIG["T_TOTAL_MS"]
    dt = CONFIG["DT_MS"]
    steps = int(T / dt)

    chain = NeuronChain(N=int(CONFIG["N_NEURONS"]))

    print("[Multi-HH Neuron Chain Simulation]")
    print("------------------------------------------------------------")
    print(f"N_NEURONS   : {chain.N}")
    print(f"T_TOTAL_MS  : {T}")
    print(f"DT_MS       : {dt}")
    print("------------------------------------------------------------")
    print(f"{'t(ms)':>8} | " +
          " | ".join([f"V{i}(mV)".rjust(8) for i in range(chain.N)]) +
          " | Events")
    print("-" * (12 + chain.N * 12))

    # 📐 간단한 로그: 몇 ms마다 한 번씩만 출력
    log_interval_ms = 5.0
    next_log_t = 0.0

    t = 0.0
    for step in range(steps):
        # 📐 스텝 진행
        chain.step(t_ms=t, dt_ms=dt)

        # 📐 로그 출력
        if t >= next_log_t - 1e-9:
            Vs = [f"{n.V:8.2f}" for n in chain.neurons]
            events = []
            for i, n in enumerate(chain.neurons):
                if n.spike:
                    events.append(f"S{i}")
            event_str = ",".join(events) if events else "-"
            print(f"{t:8.2f} | " + " | ".join(Vs) + f" | {event_str}")
            next_log_t += log_interval_ms

        t += dt

    # 📐 최종 요약
    print("\n[Summary]")
    summary = chain.summary()
    print(f"Total spikes           : {summary['total_spikes']}")
    print(f"Spikes per neuron      : {summary['spikes_per_neuron']}")
    print(f"Energy per neuron (arb):")
    for i, e in enumerate(summary["energy_per_neuron"]):
        print(f"  neuron {i}: {e:.4f}")


if __name__ == "__main__":
    run_simulation()
