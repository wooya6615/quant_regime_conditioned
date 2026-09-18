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