"""
Pilot: 국면별(regime-conditioned) XGBoost vs 국면 무시(baseline) XGBoost.

KASPER 프레임워크(Gumbel-Softmax 국면 감지 + 국면별 스플라인 KAN + 심볼릭 회귀)를
바로 구현하는 대신, "국면 조건부 모델링 자체가 XGBoost baseline보다 나은가"라는
핵심 질문만 K-means로 싸게 먼저 검증한다.

⚠️ 국면 모델(K-means)은 fold마다 train 구간에서만 새로 fit한다. 전체 데이터로
한 번 fit하면 미래 test 구간의 분포 정보가 과거 국면 라벨에 leak된다 -- 이 프로젝트
learnings.md의 "look-ahead leakage patterns"와 동일한 함정이므로 반드시 피할 것.

전제:
    quant_xgboost/quant_position_sizing의 feature_engineering_triple_barrier.py로
    064350_features_triple_barrier_pt2sl1_nd20_hl_base.csv 가 data/ 밑에 이미
    만들어져 있어야 함 (production 채택 설정과 동일한 pt_sl=(2,1), num_days=20).

사전등록: docs/prereg_regime_conditioned_pilot.md 참고

사용법 (레포 루트에서):
    python -m src.experiments.run_regime_conditioned_pilot
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

TICKER_KRX = "064350"
CONFIG_LABEL = "pt2sl1_nd20_hl"  # production 채택 설정과 동일 (build_production_models.py 참고)

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
REGIME_FEATURE_COLS = ["hist_vol_20d", "return_20d", "bb_width", "macd_hist"]

TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, 20
SEEDS = [42, 1, 7, 123, 2024]
MIN_REGIME_ROWS = 60  # 이보다 적은 국면은 그 fold에서 전용 모델 대신 baseline으로 대체

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
)


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}_base.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 -- quant_xgboost/quant_position_sizing에서 먼저 이 config로 "
            f"triple-barrier BASE 데이터셋을 만들어야 함 (pt_sl=(2,1), num_days=20)."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


# ------------------------------------------------------------------
# 2. Walk-Forward 분할 (기존 ablation 스크립트들과 동일)
# ------------------------------------------------------------------
def walk_forward_splits(n_rows: int, train_size: int, test_size: int, step: int, embargo: int):
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = list(range(start, start + train_size))
        test_start = start + train_size + embargo
        test_idx = list(range(test_start, test_start + test_size))
        splits.append((train_idx, test_idx))
        start += step
    return splits


# ------------------------------------------------------------------
# 3. fold 안에서만 K-means fit (leakage 방지 -- 절대 전체 데이터로 fit하지 말 것)
# ------------------------------------------------------------------
def fit_fold_regimes(df: pd.DataFrame, train_idx: list, test_idx: list, seed: int):
    train_X = df[REGIME_FEATURE_COLS].iloc[train_idx]
    scaler = StandardScaler().fit(train_X)
    train_scaled = scaler.transform(train_X)

    best_km, best_score = None, -1.0
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(train_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(train_scaled, labels)
        if score > best_score:
            best_km, best_score = km, score

    all_idx = train_idx + test_idx
    all_scaled = scaler.transform(df[REGIME_FEATURE_COLS].iloc[all_idx])
    regime_labels = pd.Series(best_km.predict(all_scaled), index=range(len(all_idx)))
    return regime_labels, best_km.n_clusters


# ------------------------------------------------------------------
# 4. baseline / 국면조건부 학습·예측
# ------------------------------------------------------------------
def train_predict_baseline(df, train_idx, test_idx, seed):
    X, y = df[FEATURE_COLS_BASE], df["label_tb_binary"]
    model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    return model.predict_proba(X.iloc[test_idx])[:, 1]


def train_predict_regime_conditioned(df, train_idx, test_idx, seed):
    """fold 안에서 국면별로 쪼개서 학습. 표본이 MIN_REGIME_ROWS 미만인 국면은
    baseline(전체 train)으로 fallback -- 국면 하나가 데이터를 과도하게 쪼개
    과적합시키는 것을 막기 위함."""
    X, y = df[FEATURE_COLS_BASE], df["label_tb_binary"]
    regime_labels, n_regimes = fit_fold_regimes(df, train_idx, test_idx, seed)

    train_regimes = regime_labels.iloc[: len(train_idx)].values
    test_regimes = regime_labels.iloc[len(train_idx):].values

    fallback_model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    fallback_model.fit(X.iloc[train_idx], y.iloc[train_idx])

    regime_models = {}
    for r in range(n_regimes):
        r_train_idx = [train_idx[i] for i, rl in enumerate(train_regimes) if rl == r]
        if len(r_train_idx) < MIN_REGIME_ROWS:
            continue
        m = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        m.fit(X.iloc[r_train_idx], y.iloc[r_train_idx])
        regime_models[r] = m

    proba = np.zeros(len(test_idx))
    for i, (row_idx, r) in enumerate(zip(test_idx, test_regimes)):
        model = regime_models.get(r, fallback_model)
        proba[i] = model.predict_proba(X.iloc[[row_idx]])[:, 1][0]
    return proba, n_regimes


# ------------------------------------------------------------------
# 5. seed 하나 전체 실행
# ------------------------------------------------------------------
def run_seed(df: pd.DataFrame, seed: int) -> dict:
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)
    y_all = df["label_tb_binary"].values

    baseline_auc, regime_auc, n_regime_list = [], [], []
    for train_idx, test_idx in splits:
        y_test = y_all[test_idx]
        if len(set(y_test)) < 2:
            continue

        base_proba = train_predict_baseline(df, train_idx, test_idx, seed)
        regime_proba, n_regimes = train_predict_regime_conditioned(df, train_idx, test_idx, seed)

        baseline_auc.append(roc_auc_score(y_test, base_proba))
        regime_auc.append(roc_auc_score(y_test, regime_proba))
        n_regime_list.append(n_regimes)

    return {
        "seed": seed,
        "n_folds": len(baseline_auc),
        "baseline_auc_mean": float(np.mean(baseline_auc)),
        "regime_auc_mean": float(np.mean(regime_auc)),
        "auc_diff": float(np.mean(regime_auc) - np.mean(baseline_auc)),
        "avg_n_regimes": float(np.mean(n_regime_list)),
    }


# ------------------------------------------------------------------
# 6. 메인
# ------------------------------------------------------------------
def main():
    print(f"=== 국면 조건부(regime-conditioned) XGBoost pilot: {TICKER_KRX} ===\n")
    df = load_dataset()
    print(f"데이터 {len(df)}행 로드 완료 ({df.index.min().date()} ~ {df.index.max().date()})\n")

    results = [run_seed(df, seed) for seed in SEEDS]
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    n_pass = int((result_df["auc_diff"] > 0).sum())
    diff_mean = result_df["auc_diff"].mean()
    diff_std = result_df["auc_diff"].std()

    print(f"\n{n_pass}/{len(SEEDS)} 시드에서 국면 조건부 모델이 baseline AUC를 앞섬")
    print(f"AUC 차이 평균: {diff_mean:+.4f} (표준편차 {diff_std:.4f}, "
          f"std/mean = {abs(diff_std / diff_mean):.1%})" if diff_mean != 0 else "")

    if n_pass < 4:
        print(
            "\n판정: [1차 스크리닝 실패] -- AUC 단계에서부터 5-seed 중 4개 이상 개선을 "
            "못 넘음. Gumbel-Softmax/KAN으로 확장할 근거 부족.\n"
            "GRU/TCN [실패]와 같은 결론: 데이터 규모 대비 모델을 더 쪼개는 방향은 "
            "이 종목/프레임에서 안 통함. KASPER 탐색 라인 종료 권장."
        )
    else:
        print(
            "\n판정: [1차 스크리닝 통과] -- 다음 단계로 국면 배제 재검증(regime exclusion) "
            "+ 실제 backtest(비용 반영 net_return)로 진행 필요.\n"
            "⚠️ AUC 개선이 실전 손익으로 이어진다는 보장은 없음 (learnings.md: "
            "'AUC 개선 ≠ 실전 손익 개선'). 여기서 멈추고 KAN을 얹지 말 것."
        )


if __name__ == "__main__":
    main()