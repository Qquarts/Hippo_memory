# 🧠 HIPPO_DREAM_v1

**Hippocampus-Cortex Memory Consolidation System**

> Wake → Sleep → Wake 메모리 공고화 시스템  
> 지은이: GNJz | 발행: 2025.11.24

---

## 📖 Overview

생물학적으로 타당한 해마-피질 메모리 공고화 모델 구현

**핵심 개념**:
- **Wake Phase**: 해마가 새로운 패턴을 빠르게 학습
- **Sleep Phase**: 꿈(replay)을 통해 피질로 메모리 전송
- **Wake Phase**: 피질이 장기 기억으로 회상

---

## 🚀 Quick Start

```bash
# 실행
python3 hippo_dream.py

# 예상 출력:
# ✅ Hippocampus → Cortex: 3/3
# ✅ Direct Cortex Recall: 3/3
# 🎉 Perfect Consolidation!
```

---

## 📐 구현된 핵심 수식

### 1️⃣ Hodgkin-Huxley Neuron Dynamics
```
C_m dV/dt = I_ext + I_syn - g_L(V-E_L) - g_Na·m³h(V-E_Na) - g_K·n⁴(V-E_K)

Gating variables:
dm/dt = α_m(1-m) - β_m·m
dh/dt = α_h(1-h) - β_h·h
dn/dt = α_n(1-n) - β_n·n
```

### 2️⃣ Short-Term Plasticity (STP/PTP)
```
On spike:  S ← S + 0.3,    PTP ← PTP + 0.05
Decay:     S ← S - 0.01,   PTP ← PTP - 0.001
```

### 3️⃣ Subiculum Integration
```
y(t+dt) = (1-α)·y(t) + spike(t)
where α = dt/τ
```

### 4️⃣ Cortex Ridge Regression
```
Training: W = Y·X^T·(X·X^T + αI)^(-1)
Inference: p_i = exp(z_i) / Σ_j exp(z_j)
```

### 5️⃣ Incremental Learning (Sleep)
```
error = y - ŷ
W ← W + η·(error ⊗ input)
```

### 6️⃣ Hippocampal Replay
```
I_DG = I_base + N(0, σ)
Q_ij ← Q_ij + f(S, PTP)
```

### 7️⃣ Synaptic Decay
```
Q_max ← Q_max · decay_rate
```

---

## 🏗️ Network Architecture

```
Phase 1 - Wake (Learning):
  DG → CA3 (clusters) → CA1 → Subiculum → Cortex
       ↻ recurrent

Phase 2 - Sleep (Consolidation):
  1. Hippocampal Replay
  2. Cortical Consolidation
  3. Hippocampal Decay

Phase 3 - Wake (Recall):
  - Hippocampus → Cortex (약화된 해마)
  - Direct Cortex (해마 우회)
```

---

## 📦 Dependencies

이 모듈은 `v3_event.py`에 의존합니다:
- **CONFIG**: Global Hodgkin-Huxley parameters
- **HHSomaQuick**: Fast HH soma implementation
- **SynapseCore**: Synaptic event engine with delay queue

**Python packages**:
- `numpy`: 수치 계산
- `random`: 난수 생성

---

## 🔬 생물학적 근거

### Systems Consolidation Theory
- 해마는 단기 저장소 (weeks~months)
- 수면 중 replay를 통해 피질로 전송
- 피질은 장기 저장소 (years~lifetime)

### Memory Replay during Sleep
- 해마가 낮의 경험을 재생
- 약한 자극 + 노이즈 = 자발적 활성화
- Replay를 통해 시냅스 강화

### Hippocampal-Neocortical Dialogue
- 해마 replay → 피질 점진 학습
- 느린 피질 학습 → 안정적 장기 저장
- 해마 약화 → 피질 의존 증가

---

## 📊 테스트 결과

```
Phase 1: Wake Learning
  ✅ Pattern A: [0, 1, 2] → Stored
  ✅ Pattern B: [6, 7, 8] → Stored
  ✅ Pattern C: [12, 13, 14] → Stored

Phase 2: Sleep Consolidation
  🌀 3 Dream Cycles
  🧠 Cortex: 9 replays consolidated
  🔻 Hippocampus: 30% decay

Phase 3: Wake Recall
  ✅ Hippocampus → Cortex: 3/3
  ✅ Direct Cortex: 3/3
  🎉 Perfect Consolidation!
```

---

## 🎓 활용 분야

### 교육
- 신경과학: Systems Consolidation Theory
- 수리 생물학: HH dynamics, STP/PTP
- 기계학습: Ridge Regression, Incremental Learning

### 연구
- Memory consolidation 연구
- Sleep function 연구
- Hippocampal-cortical interaction

---

## 📚 주요 파라미터

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `I_base` | 50.0 | Replay 자극 (wake: 200.0) |
| `noise_level` | 0.3 | Replay 노이즈 |
| `decay_rate` | 0.7 | 해마 감쇠율 (30% 약화) |
| `lr` | 0.03 | 피질 학습률 |
| `num_replays` | 3 | Sleep 반복 횟수 |

---

## 📄 파일 구조

```
hippo_dream.py (884 lines)
├── LightNeuron: HH + STP/PTP
├── SubiculumFast: Low-pass filter
├── CortexRidge: Ridge + Incremental
├── apply_wta: Winner-take-all
├── hippocampal_replay: Sleep replay
├── synaptic_decay: Hippocampal decay
├── cortex_consolidation: Cortical learning
└── run_dream_simulation: Main pipeline
```

---

## 🔍 코드 읽기 가이드

1. **파일 헤더** (line 1-62): 전체 수식 요약
2. **LightNeuron** (line 67-108): 기본 뉴런
3. **SubiculumFast** (line 113-148): 단기 통합
4. **CortexRidge** (line 151-274): 장기 학습
5. **Dream Functions** (line 304-440): 공고화
6. **Main Simulation** (line 442-884): 전체

---

## ✅ 검증 완료

- ✅ 모든 수식 검증 완료
- ✅ 생물학적 타당성 확인
- ✅ 3/3 패턴 완벽 회상
- ✅ Consolidation 성공

---

## 📞 Contact

**Author**: GNJz  
**Date**: 2025.11.24  
**Version**: HIPPO_DREAM_v1

---

## 📜 License

Research and Educational Use

---

**🧠 "수면은 뇌의 정리 시간이다" — Systems Consolidation Theory**
