"""
Pilot v2: 국면별(regime-conditioned) XGBoost vs baseline -- 064350 + 118990 풀링.

064350 단독 실행에서 5/5 시드 전부 음수로 [실패] 판정됐고, avg_n_regimes~2.3에서
국면당 표본이 ~150행까지 쪼개지는 게 원인으로 진단됨. 표본 부족이 진짜 원인인지
확인하기 위해 모트렉스(118990)를 풀링해서 재검증한다.

⚠️ 두 종목은 상장일이 다르므로, 기존 단독 종목 스크립트처럼 "행 번호" 기준으로
walk-forward를 나누면 두 종목의 시점이 어긋나 미래 데이터가 과거로 샐 수 있다.
이 스크립트는 **날짜 기준**으로 fold 경계를 정하고, 각 fold 안에서 두 종목의
해당 날짜 구간 행을 모아 학습한다 (row-position 기준 아님).

⚠️ `ticker` 식별 feature는 추가하지 않는다 (quant_ranking_kr의 "ticker feature
의존/fold 불안정성 심화로 개선 없음" 기록에 따름). 순수하게 표본 수만 늘림.

⚠️ K-means도 여전히 fold의 pooled train 구간에서만 fit한다 (look-ahead leakage 방지).

사전등록: docs/prereg_regime_conditioned_pilot.md의 "Addendum" 절 참고

사용법 (레포 루트에서):
    python -m src.experiments.run_regime_conditioned_pilot_pooled
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

CONFIG_LABEL = "pt2sl1_nd20_hl"  # production 3종목 풀링과 동일 설정
TICKERS = {
    "064350": "현대로템",
    "118990": "모트렉스",
}

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
REGIME_FEATURE_COLS = ["hist_vol_20d", "return_20d", "bb_width", "macd_hist"]

TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, 20  # 거래일(calendar position) 기준
SEEDS = [42, 1, 7, 123, 2024]
MIN_REGIME_ROWS = 60

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
)


# ------------------------------------------------------------------
# 1. 데이터 로드 -- 종목별로 따로 읽고, 교집합 구간으로 자름
# ------------------------------------------------------------------
def load_ticker_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_base.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 -- quant_xgboost/quant_position_sizing에서 이 config로 "
            f"triple-barrier BASE 데이터셋을 만들어야 함 (pt_sl=(2,1), num_days=20)."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def load_pooled() -> dict:
    dfs = {tk: load_ticker_dataset(tk) for tk in TICKERS}
    overlap_start = max(df.index.min() for df in dfs.values())
    overlap_end = min(df.index.max() for df in dfs.values())
    for tk in dfs:
        before = len(dfs[tk])
        dfs[tk] = dfs[tk].loc[overlap_start:overlap_end]
        print(f"  {tk} ({TICKERS[tk]}): {before}행 -> 교집합 구간 {len(dfs[tk])}행 "
              f"({overlap_start.date()} ~ {overlap_end.date()})")
    return dfs


# ------------------------------------------------------------------
# 2. 날짜 기준 walk-forward 분할 (두 종목 공통 calendar 위에서)
# ------------------------------------------------------------------
def build_calendar(dfs: dict) -> pd.DatetimeIndex:
    all_dates = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    return pd.DatetimeIndex(all_dates)


def date_walk_forward_splits(calendar: pd.DatetimeIndex, train_size, test_size, step, embargo):
    splits = []
    start = 0
    n = len(calendar)
    while start + train_size + embargo + test_size <= n:
        train_dates = calendar[start: start + train_size]
        test_start_pos = start + train_size + embargo
        test_dates = calendar[test_start_pos: test_start_pos + test_size]
        splits.append((train_dates, test_dates))
        start += step
    return splits


def pooled_rows(dfs: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    parts = [df[df.index.isin(dates)] for df in dfs.values()]
    return pd.concat(parts, axis=0)


# ------------------------------------------------------------------
# 3. fold 안에서만 K-means fit (pooled train 구간, leakage 방지)
# ------------------------------------------------------------------
def fit_fold_regimes(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int):
    train_X = train_df[REGIME_FEATURE_COLS]
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

    train_regimes = pd.Series(best_km.predict(train_scaled), index=train_df.index)
    test_scaled = scaler.transform(test_df[REGIME_FEATURE_COLS])
    test_regimes = pd.Series(best_km.predict(test_scaled), index=test_df.index)
    return train_regimes, test_regimes, best_km.n_clusters


# ------------------------------------------------------------------
# 4. baseline / 국면조건부 학습·예측 (pooled)
# ------------------------------------------------------------------
def train_predict_baseline(train_df, test_df, seed):
    X_train, y_train = train_df[FEATURE_COLS_BASE], train_df["label_tb_binary"]
    X_test = test_df[FEATURE_COLS_BASE]
    model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def train_predict_regime_conditioned(train_df, test_df, seed):
    X_train, y_train = train_df[FEATURE_COLS_BASE], train_df["label_tb_binary"]
    X_test = test_df[FEATURE_COLS_BASE]
    train_regimes, test_regimes, n_regimes = fit_fold_regimes(train_df, test_df, seed)

    fallback_model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    fallback_model.fit(X_train, y_train)

    regime_models = {}
    for r in range(n_regimes):
        r_mask = train_regimes.values == r
        if r_mask.sum() < MIN_REGIME_ROWS:
            continue
        m = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        m.fit(X_train[r_mask], y_train[r_mask])
        regime_models[r] = m

    proba = np.zeros(len(test_df))
    for i, r in enumerate(test_regimes.values):
        model = regime_models.get(r, fallback_model)
        proba[i] = model.predict_proba(X_test.iloc[[i]])[:, 1][0]
    return proba, n_regimes


# ------------------------------------------------------------------
# 5. seed 하나 전체 실행
# ------------------------------------------------------------------
def run_seed(dfs: dict, calendar: pd.DatetimeIndex, seed: int) -> dict:
    splits = date_walk_forward_splits(calendar, TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    baseline_auc, regime_auc, n_regime_list, pool_sizes = [], [], [], []
    for train_dates, test_dates in splits:
        train_df = pooled_rows(dfs, train_dates)
        test_df = pooled_rows(dfs, test_dates)
        y_test = test_df["label_tb_binary"].values
        if len(set(y_test)) < 2 or train_df.empty or test_df.empty:
            continue

        base_proba = train_predict_baseline(train_df, test_df, seed)
        regime_proba, n_regimes = train_predict_regime_conditioned(train_df, test_df, seed)

        baseline_auc.append(roc_auc_score(y_test, base_proba))
        regime_auc.append(roc_auc_score(y_test, regime_proba))
        n_regime_list.append(n_regimes)
        pool_sizes.append(len(train_df))

    return {
        "seed": seed,
        "n_folds": len(baseline_auc),
        "avg_train_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "baseline_auc_mean": float(np.mean(baseline_auc)),
        "regime_auc_mean": float(np.mean(regime_auc)),
        "auc_diff": float(np.mean(regime_auc) - np.mean(baseline_auc)),
        "avg_n_regimes": float(np.mean(n_regime_list)),
    }


# ------------------------------------------------------------------
# 6. 메인
# ------------------------------------------------------------------
def main():
    tickers_str = " + ".join(f"{tk}({name})" for tk, name in TICKERS.items())
    print(f"=== 국면 조건부 XGBoost pilot (풀링): {tickers_str} ===\n")

    dfs = load_pooled()
    calendar = build_calendar(dfs)
    print(f"\n공통 calendar: {len(calendar)}거래일 ({calendar.min().date()} ~ {calendar.max().date()})\n")

    results = [run_seed(dfs, calendar, seed) for seed in SEEDS]
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    n_pass = int((result_df["auc_diff"] > 0).sum())
    diff_mean = result_df["auc_diff"].mean()
    diff_std = result_df["auc_diff"].std()

    print(f"\n{n_pass}/{len(SEEDS)} 시드에서 국면 조건부 모델이 baseline AUC를 앞섬")
    if diff_mean != 0:
        print(f"AUC 차이 평균: {diff_mean:+.4f} (표준편차 {diff_std:.4f}, "
              f"std/mean = {abs(diff_std / diff_mean):.1%})")

    if n_pass < 4:
        print(
            "\n판정: [실패] (풀링 확장에서도 실패) -- 표본 부족이 원인이 아니라 "
            "'국면 조건부 모델링 자체가 이 문제 설정에서 안 통한다'는 더 강한 결론.\n"
            "KASPER 탐색 라인 완전 종료 권장. 재시도 불필요."
        )
    else:
        print(
            "\n판정: [1차 스크리닝 통과] (풀링 확장) -- 표본 부족이 진짜 원인이었을 "
            "가능성. 3종목(한전기술 포함) 전체 풀링으로 재확인 후 "
            "국면 배제 재검증 + 실제 backtest로 진행.\n"
            "⚠️ 여전히 AUC 단계일 뿐, 실전 손익 검증 전까지 KAN을 얹지 말 것."
        )


if __name__ == "__main__":
    main()