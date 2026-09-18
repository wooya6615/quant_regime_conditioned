# 사전등록: 국면 조건부(regime-conditioned) XGBoost pilot

- 레포(제안): `quant_regime_conditioned`
- 브랜치(제안): `experiment/regime-conditioned-xgboost-pilot`
- 작성일: 2026-09-18
- 배경: KASPER 프레임워크(Gumbel-Softmax 국면 감지 + 국면별 스플라인 KAN + 심볼릭 회귀) 검토.
  전체를 바로 구현하기엔 비용이 크고, `quant_unsupervised`의 K-means 국면 분석(모트렉스
  7.76배, 한전기술 5.08배 이례성 확인)이 이미 "국면에 따라 수익 구조가 달라진다"는
  전제를 뒷받침하므로, KAN/Gumbel-Softmax 없이 **국면 조건부 모델링 자체가 통하는가**만
  XGBoost로 싸게 먼저 검증한다.

## 1. 가설 / 질문

같은 BASE feature(13개)·같은 종목·같은 triple-barrier walk-forward 프레임에서,
"국면(K-means)마다 별도 XGBoost를 학습"하는 게 "국면을 무시한 단일 XGBoost"보다
예측력이 나은가?

## 2. 고정할 파라미터 (사전등록 -- 결과 보고 바꾸지 않음)

| 파라미터 | 값 |
|---|---|
| 종목 | 064350 (production 채택 종목, 데이터 가장 안정적) |
| feature set | FEATURE_COLS_BASE (13개, 밸류에이션 등 alternative data 제외) |
| 라벨 | `label_tb_binary` (pt_sl=(2,1), num_days=20 -- production 풀링 설정과 동일) |
| walk-forward | train_size=300, test_size=60, step=60, embargo=20 |
| 국면 feature | REGIME_FEATURE_COLS = hist_vol_20d, return_20d, bb_width, macd_hist
  (`quant_unsupervised`와 동일 4개 -- FEATURE_COLS_BASE에 이미 포함돼 있어 별도 계산 불요) |
| 국면 모델 | K-means, K=2~6 중 silhouette 최적 -- **fold의 train 구간에서만 fit**
  (fold 밖 데이터 사용 금지 -- look-ahead leakage 방지) |
| 국면별 최소 표본 | 60행 미만인 국면은 그 fold에서 전용 모델을 만들지 않고 baseline(전체 train)으로 fallback |
| seeds | 42, 1, 7, 123, 2024 |
| 1차 평가지표 | fold별 AUC (baseline vs 국면조건부), 5-seed 비교 |

## 3. 통과 기준 (1차 스크리닝 -- 이 단계는 AUC만 봄)

- 5-seed 중 4개 이상에서 `regime_auc_mean > baseline_auc_mean`
- std/mean < 50% (기존 판정 엄격도와 동일)

## 4. 이 pilot에서 하지 않는 것

- Gumbel-Softmax(학습 가능한 국면 감지)로 바꾸지 않음 -- K-means 그대로 재사용
- KAN/스플라인/심볼릭 회귀 구현하지 않음 -- 국면 조건부 자체의 유효성만 확인
- 실전 backtest(비용 반영 net_return)는 이 단계에서 하지 않음 -- `learnings.md`의
  "AUC 개선 ≠ 실전 손익 개선" 원칙에 따라, 1차 통과 시에만 다음 단계로 진행

## 5. 알려진 리스크 (미리 인지)

- 국면별로 쪼개면 이미 부족한 데이터(300행 train)가 더 쪼개짐 -- GRU/TCN이 데이터
  부족으로 실패했던 것과 같은 함정. MIN_REGIME_ROWS=60 fallback으로 최소한의 안전장치만 둠.
- fold마다 K-means를 다시 fit하므로 국면 개수(K)와 국면의 "의미"가 fold마다 달라질 수
  있음 -- 이건 leakage 방지를 위해 감수하는 트레이드오프. 국면에 해석 가능한 이름을
  붙이는 건(KASPER의 핵심 가치) 이 pilot 통과 이후, 전체 데이터로 1회 최종 fit할 때 논의.

## 6. 판정 후 처리

- 1차 통과 (4/5 이상) -> 다음 단계: 국면 배제 재검증(regime exclusion) + 실제 backtest로
  진행. 그 다음에야 Gumbel-Softmax/KAN 확장 검토.
- 1차 실패 -> `quant_seq_model`의 GRU/TCN [실패]와 같은 결론(데이터 규모 대비 모델을
  더 쪼개는 방향은 안 통함)으로 기록하고, KASPER 탐색 라인 종료.

## 7. 결과 (064350 단독, 2026-09-18)

5/5 시드 전부 음수 (AUC 차이 평균 -0.0109, std/mean 30.5%). `avg_n_regimes`가
거의 항상 2.317로 나와 fold마다 K=2가 최적으로 뽑혔고, train 300행이 국면당
~150행으로 쪼개짐. GRU/TCN [실패]와 동일한 "데이터 규모 대비 모델/구조를 더
쪼개는 방향은 안 통함" 패턴 재현.

**판정: [실패]** (064350 단독 기준)

## Addendum (2026-09-18): 종목 풀링 확장

단독 종목 실패의 원인이 "국면당 표본 부족"으로 진단됐으므로, 종료 전에 마지막으로
**모트렉스(118990) 풀링**으로 표본을 늘렸을 때도 같은 결론인지 확인한다. 이건 새
가설이 아니라 같은 가설(국면 조건부 XGBoost가 baseline보다 나은가)을 더 큰 표본에서
재검증하는 것이므로, 원래 사전등록을 폐기하지 않고 addendum으로 남김 (`postmortem`
성격의 재실행이 아니라, "데이터가 부족해서 실패했다"는 진단 자체를 검증하는 단계).

### 추가/변경 파라미터

| 파라미터 | 값 |
|---|---|
| 종목 | 064350 + 118990 (production 3종목 중 2개, 동일 config_label 재사용) |
| feature set / 라벨 / seeds | 1절과 동일, 변경 없음 |
| walk-forward | **날짜 기준**으로 변경 (기존 행 번호 기준은 두 종목 상장일이 달라 그대로
  이어붙이면 시점이 어긋남 -- 두 종목의 거래일 교집합 구간에서 calendar 기준 300/60/60/20) |
| ticker 식별 feature | 추가하지 않음 -- `quant_ranking_kr`의 "ticker feature 의존/fold
  불안정성 심화로 개선 없음" 기록을 따라, 순수하게 표본 수만 늘리고 모델 입력은 그대로 |
| 국면 모델 | 동일 (fold의 pooled train 구간에서만 fit) -- 이제 두 종목이 같이 들어가므로
  "종목별 특이 패턴"이 아니라 "시장 전반의 국면"에 더 가까워질 것으로 기대 |

### 통과 기준

1절과 동일 (5-seed 중 4/5 이상, std/mean < 50%).

### 판정 후 처리

- 통과 -> 표본 부족이 진짜 원인이었다는 뜻 -> 3종목(한전기술 포함) 전체 풀링 +
  국면 배제 검증으로 확장
- 실패 -> 표본 부족이 아니라 "국면 조건부 모델링 자체가 이 문제 설정에서 안 통한다"는
  더 강한 결론 -> KASPER 탐색 라인 완전 종료, 재시도 없음