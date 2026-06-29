import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import time

# ============================================================
# 日経225先物｜反発足・ブレイク順張りアプリ
# 参照データ: Yahoo Finance NIY=F（CME Nikkei/Yen Futures）
# 実発注前は必ずマネックスの日経225マイクロの価格・チャートで確認する。
# ============================================================

st.set_page_config(
    page_title="日経225先物｜反発ブレイク順張り",
    page_icon="📈",
    layout="wide",
)

APP_VERSION = "戦略版 v5（必須条件＋加点1つ以上）"
TICKER_INTRADAY = "NIY=F"
TICKER_DAILY = "^N225"
JST = "Asia/Tokyo"

# 日経225マイクロ前提
TICK = 5
YEN_PER_POINT_PER_MICRO = 10
RISK_PER_TRADE_PCT = 0.01
STOP_BUFFER = 5
TARGET_R1 = 1.5
TARGET_R2 = 2.0

# 参照範囲
RANGE_REFERENCE_BARS = 108       # 約9時間：重要レンジ・抵抗帯/支持帯の参照
PULLBACK_CONTEXT_BARS = 3        # 反発足の前に押し/戻しがあったか
STRUCTURE_STOP_BARS = 8          # 約40分：直近押し安値/戻り高値の参照
RECENT_BREAKOUT_BARS = 12        # 約1時間：直近高安の明確ブレイク判定
PIVOT_LEFT_RIGHT = 2             # 抵抗帯/支持帯の簡易スイング判定
BACKTEST_HOLD_BARS = 36          # 約3時間で決済評価
BACKTEST_COOLDOWN_BARS = 6

# 必須条件①：レンジ除外
TREND15_SLOPE_LOOKBACK = 6       # 15分足6本＝約90分
TREND15_CLEAR_SLOPE_ATR = 0.55   # EMA変化が15分ATRの0.55倍以上で明確な傾き
TREND15_FLAT_SLOPE_ATR = 0.35    # 0.35倍未満は横ばい扱い
TREND15_CROSS_LOOKBACK = 8       # 約2時間
TREND15_MAX_SIDE_FLIPS = 1       # 2回以上MAを跨ぐとレンジ除外
FIVE_MA_TOUCH_LOOKBACK = 12      # 約1時間
FIVE_MA_MAX_TOUCH_BARS = 7       # 接触足が多すぎる場面は除外
FIVE_MA_MAX_TOUCH_EVENTS = 3     # 4回目以上の接触イベントはレンジ扱い
MA_TOUCH_TOLERANCE = TICK

# 加点条件
MAX_VALID_TOUCH_EVENT_NO = 2     # 25MAへの接触が初回〜2回目なら加点
BREAKOUT_MIN_RANGE_ATR = 0.70    # 強いブレイク足の最低値幅
BREAKOUT_MIN_BODY_RATIO = 0.55   # 強いブレイク足の最低実体比
BREAKOUT_MAX_WICK_RATIO = 0.25   # 逆行ヒゲの上限
BREAKOUT_CLOSE_POSITION = 0.70   # 高値圏/安値圏で引けたか
CLEAR_BREAK_BUFFER_ATR = 0.10    # 直近高安を「明確に」抜く余裕
MIN_BONUS_SCORE = 1              # 加点条件は1つ以上でエントリー可


# ------------------------------------------------------------------
# 取得・共通処理
# ------------------------------------------------------------------
def clean_columns(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_jst_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC").tz_convert(JST)
    else:
        idx = idx.tz_convert(JST)
    out.index = idx
    out.index.name = "Datetime"
    return out


@st.cache_data(ttl=90, show_spinner=False)
def load_intraday(period: str = "5d") -> pd.DataFrame:
    data = yf.download(
        TICKER_INTRADAY,
        period=period,
        interval="5m",
        auto_adjust=False,
        prepost=True,
        progress=False,
        threads=False,
    )
    df = clean_columns(data)
    required = {"Open", "High", "Low", "Close"}
    if df.empty or not required.issubset(df.columns):
        raise RuntimeError("5分足を取得できませんでした。市場休場・Yahoo側の遅延を確認してください。")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Volume"] = df["Volume"].fillna(0).clip(lower=0)
    return to_jst_index(df).sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily() -> pd.DataFrame:
    data = yf.download(
        TICKER_DAILY,
        period="10y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    df = clean_columns(data)
    required = {"High", "Low", "Close"}
    if df.empty or not required.issubset(df.columns):
        raise RuntimeError("日足データを取得できませんでした。")
    return df.dropna(subset=["High", "Low", "Close"]).copy()


def ceil_tick(value: float) -> int:
    return int(np.ceil(float(value) / TICK) * TICK)


def floor_tick(value: float) -> int:
    return int(np.floor(float(value) / TICK) * TICK)


def get_session_kind(ts: pd.Timestamp) -> str:
    t = ts.tz_convert(JST).time() if ts.tzinfo else ts.time()
    if time(8, 45) <= t <= time(15, 45):
        return "日中"
    if t >= time(17, 0) or t <= time(6, 0):
        return "夜間"
    return "時間外"


def get_trade_date(ts: pd.Timestamp):
    local = ts.tz_convert(JST) if ts.tzinfo else ts
    return (local + pd.Timedelta(hours=7)).date()


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SESSION_KIND"] = [get_session_kind(ts) for ts in out.index]
    out["TRADE_DATE"] = [get_trade_date(ts) for ts in out.index]
    out["SESSION_ID"] = out["TRADE_DATE"].astype(str) + "_" + out["SESSION_KIND"].astype(str)
    return out


# ------------------------------------------------------------------
# 指標
# ------------------------------------------------------------------
def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["SMA25"] = close.rolling(25).mean()
    out["SMA75"] = close.rolling(75).mean()
    out["EMA12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA26"] = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    out["WEEK_SCORE"] = (
        np.where(close > out["SMA25"], 1, -1)
        + np.where(out["SMA25"] > out["SMA75"], 1, -1)
        + np.where(out["MACD"] > out["MACD_SIGNAL"], 1, -1)
        + np.select(
            [(out["RSI"] >= 52) & (out["RSI"] <= 70), out["RSI"] < 48],
            [1, -1],
            default=0,
        )
    )
    out["WEEK_SIGNAL"] = np.select(
        [out["WEEK_SCORE"] >= 3, out["WEEK_SCORE"] <= -3],
        ["買い優勢", "売り優勢"],
        default="見送り",
    )
    return out.dropna(subset=["SMA75", "MACD_SIGNAL", "RSI"]).copy()


def build_15m_trend(df5: pd.DataFrame) -> pd.DataFrame:
    bars15 = (
        df5[["Open", "High", "Low", "Close"]]
        .resample("15min", label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    bars15["EMA25_15"] = bars15["Close"].ewm(span=25, adjust=False).mean()

    prev_close = bars15["Close"].shift(1)
    tr15 = pd.concat(
        [
            bars15["High"] - bars15["Low"],
            (bars15["High"] - prev_close).abs(),
            (bars15["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    bars15["ATR14_15"] = tr15.rolling(14).mean()

    bars15["EMA25_CHANGE_N"] = bars15["EMA25_15"] - bars15["EMA25_15"].shift(TREND15_SLOPE_LOOKBACK)
    bars15["CLOSE_CHANGE_N"] = bars15["Close"] - bars15["Close"].shift(TREND15_SLOPE_LOOKBACK)
    bars15["EMA25_FLAT_15"] = (
        bars15["EMA25_CHANGE_N"].abs() < bars15["ATR14_15"] * TREND15_FLAT_SLOPE_ATR
    )

    close_side = pd.Series(
        np.select(
            [
                bars15["Close"] > bars15["EMA25_15"] + MA_TOUCH_TOLERANCE,
                bars15["Close"] < bars15["EMA25_15"] - MA_TOUCH_TOLERANCE,
            ],
            [1, -1],
            default=0,
        ),
        index=bars15.index,
    ).replace(0, np.nan).ffill().fillna(0)
    bars15["CLOSE_SIDE_15"] = close_side
    prev_side = close_side.shift(1)
    bars15["SIDE_FLIP_15"] = (
        (close_side != prev_side) & (close_side != 0) & (prev_side != 0)
    ).astype(int)
    bars15["SIDE_FLIP_COUNT_15"] = bars15["SIDE_FLIP_15"].rolling(
        TREND15_CROSS_LOOKBACK,
        min_periods=TREND15_CROSS_LOOKBACK,
    ).sum()

    bars15["TREND15"] = np.select(
        [
            (bars15["EMA25_CHANGE_N"] >= bars15["ATR14_15"] * TREND15_CLEAR_SLOPE_ATR)
            & (bars15["Close"] > bars15["EMA25_15"])
            & (bars15["CLOSE_CHANGE_N"] > 0),
            (bars15["EMA25_CHANGE_N"] <= -bars15["ATR14_15"] * TREND15_CLEAR_SLOPE_ATR)
            & (bars15["Close"] < bars15["EMA25_15"])
            & (bars15["CLOSE_CHANGE_N"] < 0),
        ],
        ["明確上昇", "明確下降"],
        default="不明確",
    )
    bars15["TREND15_CHOP_EXCLUDED"] = (
        bars15["EMA25_FLAT_15"]
        | (bars15["SIDE_FLIP_COUNT_15"] > TREND15_MAX_SIDE_FLIPS)
    )
    return bars15


def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = add_session_columns(df)
    close = out["Close"]
    out["EMA9_5"] = close.ewm(span=9, adjust=False).mean()
    out["EMA25_5"] = close.ewm(span=25, adjust=False).mean()
    out["MACD5"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["MACD5_SIGNAL"] = out["MACD5"].ewm(span=9, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR14_5"] = tr.rolling(14).mean()

    if out["Volume"].sum() > 0:
        typical = (out["High"] + out["Low"] + out["Close"]) / 3
        pv = (typical * out["Volume"]).groupby(out["SESSION_ID"]).cumsum()
        vol = out["Volume"].groupby(out["SESSION_ID"]).cumsum()
        out["VWAP"] = pv / vol.replace(0, np.nan)
    else:
        out["VWAP"] = np.nan

    # 25MA接触：連続する接触足は1イベントとして数える
    out["TOUCH_5"] = (
        (out["Low"] <= out["EMA25_5"] + MA_TOUCH_TOLERANCE)
        & (out["High"] >= out["EMA25_5"] - MA_TOUCH_TOLERANCE)
    )
    out["TOUCH_EVENT_5"] = out.groupby("SESSION_ID")["TOUCH_5"].transform(
        lambda s: s.astype(bool) & ~s.astype(bool).shift(1, fill_value=False)
    )
    out["TOUCH_EVENT_NO_5"] = out.groupby("SESSION_ID")["TOUCH_EVENT_5"].cumsum().astype(int)
    out["TOUCH_BAR_COUNT_5"] = out.groupby("SESSION_ID")["TOUCH_5"].transform(
        lambda s: s.astype(int).rolling(FIVE_MA_TOUCH_LOOKBACK, min_periods=1).sum()
    )
    out["TOUCH_EVENT_COUNT_5"] = out.groupby("SESSION_ID")["TOUCH_EVENT_5"].transform(
        lambda s: s.astype(int).rolling(FIVE_MA_TOUCH_LOOKBACK, min_periods=1).sum()
    )
    out["FIVE_MIN_CHOP_EXCLUDED"] = (
        (out["TOUCH_BAR_COUNT_5"] > FIVE_MA_MAX_TOUCH_BARS)
        | (out["TOUCH_EVENT_COUNT_5"] > FIVE_MA_MAX_TOUCH_EVENTS)
    )

    bars15 = build_15m_trend(out)
    out = pd.merge_asof(
        out.sort_index(),
        bars15[
            [
                "EMA25_15",
                "ATR14_15",
                "EMA25_CHANGE_N",
                "EMA25_FLAT_15",
                "SIDE_FLIP_COUNT_15",
                "TREND15_CHOP_EXCLUDED",
                "TREND15",
            ]
        ].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    out.index.name = "Datetime"
    return out.dropna(
        subset=[
            "EMA25_5", "EMA9_5", "ATR14_5", "EMA25_15", "ATR14_15",
            "SIDE_FLIP_COUNT_15", "TOUCH_BAR_COUNT_5", "TOUCH_EVENT_COUNT_5", "TREND15",
        ]
    ).copy()


# ------------------------------------------------------------------
# 必須条件・加点条件
# ------------------------------------------------------------------
def get_last_completed_index(df: pd.DataFrame) -> int:
    # 直近バーは形成中の可能性があるため1本前までを判定
    return max(0, len(df) - 2)


def get_range_exclusion_reasons(row: pd.Series) -> list[str]:
    reasons = []
    if bool(row.get("EMA25_FLAT_15", False)):
        change = abs(float(row.get("EMA25_CHANGE_N", 0.0)))
        atr = float(row.get("ATR14_15", 0.0))
        reasons.append(f"15分足25MAが横ばい（90分の変化 {change:.0f}円、15分ATR {atr:.0f}円）")

    flips = int(round(float(row.get("SIDE_FLIP_COUNT_15", 0.0))))
    if flips > TREND15_MAX_SIDE_FLIPS:
        reasons.append(f"15分足終値が25MAを直近約2時間で{flips}回跨いでいる")

    touch_bars = int(round(float(row.get("TOUCH_BAR_COUNT_5", 0.0))))
    touch_events = int(round(float(row.get("TOUCH_EVENT_COUNT_5", 0.0))))
    if touch_bars > FIVE_MA_MAX_TOUCH_BARS:
        reasons.append(f"5分足が25MAに直近約1時間で{touch_bars}本触れている")
    if touch_events > FIVE_MA_MAX_TOUCH_EVENTS:
        reasons.append(f"5分足25MAへの接触イベントが直近約1時間で{touch_events}回あり、多すぎる")
    return reasons


def get_trade_direction(row: pd.Series) -> str | None:
    # 必須条件②：15分足25MAと同じ方向だけ
    if row["TREND15"] == "明確上昇":
        return "BUY"
    if row["TREND15"] == "明確下降":
        return "SELL"
    return None


def pullback_context_ok(df: pd.DataFrame, reversal_i: int, direction: str) -> bool:
    """反発足の前に、少なくとも小さな押し/戻しがあるかを確認する。

    25MA接触そのものは必須にせず、直近の短期足がEMA9側へ戻った・
    逆色の足が出た、のどちらかを押し/戻しの最低条件にする。
    """
    start = max(0, reversal_i - PULLBACK_CONTEXT_BARS)
    prior = df.iloc[start:reversal_i]
    if prior.empty:
        return False

    if direction == "BUY":
        return bool(
            (prior["Close"] <= prior["EMA9_5"]).any()
            or (prior["Close"] < prior["Open"]).any()
        )
    return bool(
        (prior["Close"] >= prior["EMA9_5"]).any()
        or (prior["Close"] > prior["Open"]).any()
    )


def reaction_candle_ok(df: pd.DataFrame, reversal_i: int, direction: str) -> tuple[bool, str]:
    """必須条件③の前半：5分足で反発足が出たか。"""
    if reversal_i < 1:
        return False, "反発足の判定に必要な足が不足しています。"

    bar = df.iloc[reversal_i]
    prev = df.iloc[reversal_i - 1]
    bar_range = max(float(bar["High"] - bar["Low"]), float(TICK))
    body = abs(float(bar["Close"] - bar["Open"]))
    body_ratio = body / bar_range
    context_ok = pullback_context_ok(df, reversal_i, direction)

    if direction == "BUY":
        ok = (
            float(bar["Close"]) > float(bar["Open"])
            and float(bar["Close"]) > float(prev["Close"])
            and body_ratio >= 0.35
            and context_ok
        )
        return ok, "押しの後に反転陽線が出ている" if ok else "押し後の反転陽線が未完成"

    ok = (
        float(bar["Close"]) < float(bar["Open"])
        and float(bar["Close"]) < float(prev["Close"])
        and body_ratio >= 0.35
        and context_ok
    )
    return ok, "戻りの後に反転陰線が出ている" if ok else "戻り後の反転陰線が未完成"


def breakout_of_reaction_ok(reversal: pd.Series, breakout: pd.Series, direction: str) -> tuple[bool, str]:
    """必須条件③の後半：反発足高値/安値を実際に抜けたか。"""
    if direction == "BUY":
        ok = float(breakout["High"]) >= float(reversal["High"]) + TICK
        return ok, "反転陽線の高値を上抜け" if ok else "反転陽線高値の上抜け待ち"
    ok = float(breakout["Low"]) <= float(reversal["Low"]) - TICK
    return ok, "反転陰線の安値を下抜け" if ok else "反転陰線安値の下抜け待ち"


def build_structural_stop(
    direction: str,
    rationale_bar: pd.Series,
    structure_window: pd.DataFrame,
    rationale_label: str,
    structure_label: str,
) -> tuple[float, dict]:
    """必須条件④：固定幅ではなく、反発シナリオが否定される構造の外側へ置く。"""
    if structure_window.empty:
        structure_window = pd.DataFrame([rationale_bar])

    if direction == "BUY":
        candidates = [
            (rationale_label, float(rationale_bar["Low"])),
            (structure_label, float(structure_window["Low"].min())),
        ]
        basis_label, basis_price = min(candidates, key=lambda x: x[1])
        raw_stop = basis_price - STOP_BUFFER
        stop_reason = (
            f"{basis_label} {basis_price:,.0f}円の外側（{STOP_BUFFER}円下）。"
            "ここを割れると反発シナリオが否定される。"
        )
    else:
        candidates = [
            (rationale_label, float(rationale_bar["High"])),
            (structure_label, float(structure_window["High"].max())),
        ]
        basis_label, basis_price = max(candidates, key=lambda x: x[1])
        raw_stop = basis_price + STOP_BUFFER
        stop_reason = (
            f"{basis_label} {basis_price:,.0f}円の外側（{STOP_BUFFER}円上）。"
            "ここを超えると戻り売りシナリオが否定される。"
        )

    return raw_stop, {
        "stop_basis_label": basis_label,
        "stop_basis_price": basis_price,
        "stop_reason": stop_reason,
    }


def calculate_plan(
    direction: str,
    entry: float,
    stop: float,
    setup: str,
    mandatory_reasons: list[str],
    stop_meta: dict,
) -> dict:
    entry = ceil_tick(entry) if direction == "BUY" else floor_tick(entry)
    stop = floor_tick(stop) if direction == "BUY" else ceil_tick(stop)
    risk = (entry - stop) if direction == "BUY" else (stop - entry)
    if risk <= 0:
        return {"valid": False, "reason": "損切り位置がエントリー方向と矛盾しています。"}

    target1 = ceil_tick(entry + risk * TARGET_R1) if direction == "BUY" else floor_tick(entry - risk * TARGET_R1)
    target2 = ceil_tick(entry + risk * TARGET_R2) if direction == "BUY" else floor_tick(entry - risk * TARGET_R2)
    plan = {
        "valid": True,
        "direction": direction,
        "entry": int(entry),
        "stop": int(stop),
        "risk": int(risk),
        "target1": int(target1),
        "target2": int(target2),
        "setup": setup,
        "mandatory_reasons": mandatory_reasons,
        "bonus_reasons": [],
        "bonus_score": 0,
    }
    plan.update(stop_meta)
    return plan


def strong_breakout_score(bar: pd.Series, reversal: pd.Series, direction: str) -> tuple[bool, str]:
    """加点②：ブレイク足に値幅・実体・ヒゲ・終値位置の勢いがあるか。"""
    high = float(bar["High"])
    low = float(bar["Low"])
    opn = float(bar["Open"])
    close = float(bar["Close"])
    atr = max(float(bar.get("ATR14_5", 0.0)), float(TICK))
    bar_range = max(high - low, float(TICK))
    body = abs(close - opn)
    body_ratio = body / bar_range
    close_position = (close - low) / bar_range
    upper_wick_ratio = (high - max(opn, close)) / bar_range
    lower_wick_ratio = (min(opn, close) - low) / bar_range

    if direction == "BUY":
        passed = (
            close > opn
            and bar_range >= atr * BREAKOUT_MIN_RANGE_ATR
            and body_ratio >= BREAKOUT_MIN_BODY_RATIO
            and upper_wick_ratio <= BREAKOUT_MAX_WICK_RATIO
            and close_position >= BREAKOUT_CLOSE_POSITION
            and close >= float(reversal["High"]) + TICK
        )
        message = (
            f"ブレイク足が強い（値幅 {bar_range:.0f}円、実体比 {body_ratio:.0%}、終値が高値圏）"
            if passed else "ブレイク足の勢いは加点基準に届かない"
        )
        return passed, message

    passed = (
        close < opn
        and bar_range >= atr * BREAKOUT_MIN_RANGE_ATR
        and body_ratio >= BREAKOUT_MIN_BODY_RATIO
        and lower_wick_ratio <= BREAKOUT_MAX_WICK_RATIO
        and close_position <= 1 - BREAKOUT_CLOSE_POSITION
        and close <= float(reversal["Low"]) - TICK
    )
    message = (
        f"ブレイク足が強い（値幅 {bar_range:.0f}円、実体比 {body_ratio:.0%}、終値が安値圏）"
        if passed else "ブレイク足の勢いは加点基準に届かない"
    )
    return passed, message


def clear_recent_structure_score(
    df: pd.DataFrame,
    reversal_i: int,
    breakout: pd.Series,
    direction: str,
) -> tuple[bool, str]:
    """加点③：反発足の高安抜けだけでなく、直近1時間の高安も明確に抜けたか。"""
    start = max(0, reversal_i - RECENT_BREAKOUT_BARS)
    history = df.iloc[start:reversal_i]
    if len(history) < 3:
        return False, "直近高安の判定に必要な履歴が不足"

    atr = max(float(breakout.get("ATR14_5", 0.0)), float(TICK))
    buffer = max(float(TICK), atr * CLEAR_BREAK_BUFFER_ATR)
    if direction == "BUY":
        prior_high = float(history["High"].max())
        clear = float(breakout["Close"]) >= prior_high + buffer
        message = (
            f"直近約1時間高値 {prior_high:,.0f}円を終値で明確に上抜け"
            if clear else f"直近約1時間高値 {prior_high:,.0f}円の明確上抜けは未達"
        )
        return clear, message

    prior_low = float(history["Low"].min())
    clear = float(breakout["Close"]) <= prior_low - buffer
    message = (
        f"直近約1時間安値 {prior_low:,.0f}円を終値で明確に下抜け"
        if clear else f"直近約1時間安値 {prior_low:,.0f}円の明確下抜けは未達"
    )
    return clear, message


def _confirmed_pivots(window: pd.DataFrame) -> tuple[list[float], list[float]]:
    """未来データを使わず、現時点より前に確定済みの簡易スイングを抽出。"""
    highs: list[float] = []
    lows: list[float] = []
    if len(window) < PIVOT_LEFT_RIGHT * 2 + 1:
        return highs, lows

    for j in range(PIVOT_LEFT_RIGHT, len(window) - PIVOT_LEFT_RIGHT):
        segment = window.iloc[j - PIVOT_LEFT_RIGHT : j + PIVOT_LEFT_RIGHT + 1]
        h = float(window["High"].iloc[j])
        l = float(window["Low"].iloc[j])
        if h >= float(segment["High"].max()):
            highs.append(h)
        if l <= float(segment["Low"].min()):
            lows.append(l)
    return highs, lows


def next_barrier_and_room(
    df: pd.DataFrame,
    reversal_i: int,
    direction: str,
    entry: float,
    risk: float,
) -> tuple[float | None, float, str]:
    """加点④：次の抵抗帯/支持帯まで1.5R以上あるか。

    直近108本の『過去に確定した』スイング高値/安値を簡易的な抵抗帯/支持帯に使う。
    未来の足は参照しないため、バックテストでも先読みを避ける。
    """
    start = max(0, reversal_i - RANGE_REFERENCE_BARS)
    window = df.iloc[start:reversal_i].copy()
    highs, lows = _confirmed_pivots(window)

    if direction == "BUY":
        above = sorted([x for x in highs if x > entry + TICK])
        if not above:
            return None, float("inf"), "直近約9時間に1.5R未満の明確な上値抵抗帯を検出せず"
        barrier = above[0]
        room_r = (barrier - entry) / max(risk, TICK)
        return barrier, room_r, f"次の抵抗帯 {barrier:,.0f}円まで {room_r:.2f}R"

    below = sorted([x for x in lows if x < entry - TICK], reverse=True)
    if not below:
        return None, float("inf"), "直近約9時間に1.5R未満の明確な下値支持帯を検出せず"
    barrier = below[0]
    room_r = (entry - barrier) / max(risk, TICK)
    return barrier, room_r, f"次の支持帯 {barrier:,.0f}円まで {room_r:.2f}R"


def ma_touch_score(df: pd.DataFrame, reversal_i: int) -> tuple[bool, str, int]:
    """加点①：25MAへの接触が初回〜2回目か。接触は必須ではない。"""
    reversal = df.iloc[reversal_i]
    event_no = int(reversal.get("TOUCH_EVENT_NO_5", 0))
    nearby = df.iloc[max(0, reversal_i - 2) : reversal_i + 1]
    touched = bool(nearby["TOUCH_5"].any())
    passed = touched and 1 <= event_no <= MAX_VALID_TOUCH_EVENT_NO
    if passed:
        return True, f"5分足25MAへの接触が{event_no}回目（初回〜2回目）", event_no
    if touched:
        return False, f"5分足25MAへの接触は{event_no}回目（加点対象外）", event_no
    return False, "5分足25MAへの接触は確認できず（この条件は加点なし）", event_no


def assess_bonus_conditions(
    df: pd.DataFrame,
    reversal_i: int,
    breakout_i: int,
    plan: dict,
) -> dict:
    direction = plan["direction"]
    reversal = df.iloc[reversal_i]
    breakout = df.iloc[breakout_i]

    items: list[dict] = []
    passed, detail, touch_no = ma_touch_score(df, reversal_i)
    items.append({"name": "25MA接触が初回〜2回目", "passed": passed, "detail": detail})

    passed, detail = strong_breakout_score(breakout, reversal, direction)
    items.append({"name": "ブレイク足が強い", "passed": passed, "detail": detail})

    passed, detail = clear_recent_structure_score(df, reversal_i, breakout, direction)
    items.append({"name": "直近高安を明確に抜ける", "passed": passed, "detail": detail})

    barrier, room_r, room_detail = next_barrier_and_room(
        df=df,
        reversal_i=reversal_i,
        direction=direction,
        entry=float(plan["entry"]),
        risk=float(plan["risk"]),
    )
    passed = room_r >= TARGET_R1
    if np.isinf(room_r):
        room_detail += "（加点）"
    elif passed:
        room_detail += f"（{TARGET_R1:.1f}R以上で加点）"
    else:
        room_detail += f"（{TARGET_R1:.1f}R未満で加点なし）"
    items.append({"name": f"次の抵抗帯/支持帯まで{TARGET_R1:.1f}R以上", "passed": passed, "detail": room_detail})

    plan["bonus_items"] = items
    plan["bonus_score"] = sum(1 for x in items if x["passed"])
    plan["bonus_reasons"] = [x["detail"] for x in items if x["passed"]]
    plan["touch_event_no"] = touch_no
    plan["barrier"] = barrier
    plan["room_r"] = room_r
    return plan


# ------------------------------------------------------------------
# セットアップ判定
# ------------------------------------------------------------------
def build_triggered_plan_at(df: pd.DataFrame, breakout_i: int, require_bonus: bool = True) -> dict | None:
    """反発足（1本前）→高安ブレイク（現在バー）の完成セットアップを判定。"""
    if breakout_i < max(40, STRUCTURE_STOP_BARS + 2):
        return None

    reversal_i = breakout_i - 1
    reversal = df.iloc[reversal_i]
    breakout = df.iloc[breakout_i]

    # 必須①：レンジ除外（反発足・ブレイク足の両方で確認）
    range_reasons = get_range_exclusion_reasons(reversal)
    breakout_range_reasons = get_range_exclusion_reasons(breakout)
    all_range_reasons = list(dict.fromkeys(range_reasons + breakout_range_reasons))
    if all_range_reasons:
        return {"valid": False, "status": "見送り", "detail": "レンジ除外：" + " / ".join(all_range_reasons)}

    # 必須②：15分足25MAの向きと同方向だけ（反発足からブレイク足まで維持）
    direction = get_trade_direction(reversal)
    breakout_direction = get_trade_direction(breakout)
    if direction is None or breakout_direction != direction:
        return {"valid": False, "status": "見送り", "detail": "15分足25MAの方向が明確でない、またはブレイク中に方向が変化したため除外"}

    # 必須③：5分足反発足 + その高値/安値ブレイク
    reaction_ok, reaction_detail = reaction_candle_ok(df, reversal_i, direction)
    if not reaction_ok:
        return {"valid": False, "status": "待機", "detail": reaction_detail}

    break_ok, break_detail = breakout_of_reaction_ok(reversal, breakout, direction)
    if not break_ok:
        return {"valid": False, "status": "待機", "detail": break_detail}

    # 必須④：構造的損切り
    structure_start = max(0, reversal_i - STRUCTURE_STOP_BARS + 1)
    structure_window = df.iloc[structure_start : reversal_i + 1]
    if direction == "BUY":
        stop, stop_meta = build_structural_stop(
            "BUY", reversal, structure_window,
            "エントリー根拠の反転陽線安値", "直近押し安値",
        )
        entry = float(reversal["High"]) + TICK
        setup = "15分足上昇＋5分足反転陽線高値ブレイク"
        mandatory = [
            "レンジ除外を通過",
            "15分足25MAが明確に上向き",
            reaction_detail,
            break_detail,
        ]
    else:
        stop, stop_meta = build_structural_stop(
            "SELL", reversal, structure_window,
            "エントリー根拠の反転陰線高値", "直近戻り高値",
        )
        entry = float(reversal["Low"]) - TICK
        setup = "15分足下降＋5分足反転陰線安値ブレイク"
        mandatory = [
            "レンジ除外を通過",
            "15分足25MAが明確に下向き",
            reaction_detail,
            break_detail,
        ]

    plan = calculate_plan(direction, entry, stop, setup, mandatory, stop_meta)
    if not plan.get("valid"):
        return plan

    plan = assess_bonus_conditions(df, reversal_i, breakout_i, plan)
    plan.update({
        "bar_time": breakout.name,
        "reversal_time": reversal.name,
        "breakout_time": breakout.name,
        "triggered": True,
    })
    if require_bonus and plan["bonus_score"] < MIN_BONUS_SCORE:
        plan["valid"] = False
        plan["status"] = "見送り"
        plan["detail"] = "必須条件は満たすが、加点条件が0個のため見送り"
    else:
        plan["status"] = "買い発動" if direction == "BUY" else "売り発動"
    return plan


def build_waiting_plan_at(df: pd.DataFrame, reversal_i: int) -> dict | None:
    """反発足は完成したが、高値/安値ブレイク前の『準備段階』を表示する。

    これはエントリーではない。実際に反発足の高値/安値を抜くまでは発動しない。
    """
    if reversal_i < max(40, STRUCTURE_STOP_BARS + 1):
        return None
    reversal = df.iloc[reversal_i]

    range_reasons = get_range_exclusion_reasons(reversal)
    if range_reasons:
        return None
    direction = get_trade_direction(reversal)
    if direction is None:
        return None
    reaction_ok, reaction_detail = reaction_candle_ok(df, reversal_i, direction)
    if not reaction_ok:
        return None

    structure_start = max(0, reversal_i - STRUCTURE_STOP_BARS + 1)
    structure_window = df.iloc[structure_start : reversal_i + 1]
    if direction == "BUY":
        stop, stop_meta = build_structural_stop(
            "BUY", reversal, structure_window,
            "エントリー根拠の反転陽線安値", "直近押し安値",
        )
        entry = float(reversal["High"]) + TICK
        setup = "反転陽線形成｜高値上抜け待ち"
        mandatory = ["レンジ除外を通過", "15分足25MAが明確に上向き", reaction_detail]
    else:
        stop, stop_meta = build_structural_stop(
            "SELL", reversal, structure_window,
            "エントリー根拠の反転陰線高値", "直近戻り高値",
        )
        entry = float(reversal["Low"]) - TICK
        setup = "反転陰線形成｜安値下抜け待ち"
        mandatory = ["レンジ除外を通過", "15分足25MAが明確に下向き", reaction_detail]

    plan = calculate_plan(direction, entry, stop, setup, mandatory, stop_meta)
    if not plan.get("valid"):
        return None

    # 未発動のため、既に判定できる加点だけ暫定表示する。
    passed, detail, touch_no = ma_touch_score(df, reversal_i)
    barrier, room_r, room_detail = next_barrier_and_room(
        df, reversal_i, direction, float(plan["entry"]), float(plan["risk"])
    )
    room_passed = room_r >= TARGET_R1
    items = [
        {"name": "25MA接触が初回〜2回目", "passed": passed, "detail": detail},
        {"name": "ブレイク足が強い", "passed": False, "detail": "ブレイク前のため未判定"},
        {"name": "直近高安を明確に抜ける", "passed": False, "detail": "ブレイク前のため未判定"},
        {"name": f"次の抵抗帯/支持帯まで{TARGET_R1:.1f}R以上", "passed": room_passed, "detail": room_detail},
    ]
    plan.update({
        "bonus_items": items,
        "bonus_score": sum(1 for x in items if x["passed"]),
        "bonus_reasons": [x["detail"] for x in items if x["passed"]],
        "touch_event_no": touch_no,
        "barrier": barrier,
        "room_r": room_r,
        "bar_time": reversal.name,
        "reversal_time": reversal.name,
        "triggered": False,
        "status": "買い準備" if direction == "BUY" else "売り準備",
    })
    return plan


def detect_current_setup(df: pd.DataFrame, current_price: float) -> dict:
    i = get_last_completed_index(df)
    triggered = build_triggered_plan_at(df, i, require_bonus=True)
    if triggered is not None and triggered.get("valid"):
        # ブレイク済みで現在価格が発動ラインから離れたら、追いかけを警告
        plan = triggered.copy()
        if plan["direction"] == "BUY" and current_price > plan["entry"] + max(TICK, plan["risk"] * 0.5):
            plan["status"] = "追いかけ注意"
            plan["extra_warning"] = "反発足高値のブレイクはすでに発生済み。次の押し目まで待つ。"
        elif plan["direction"] == "SELL" and current_price < plan["entry"] - max(TICK, plan["risk"] * 0.5):
            plan["status"] = "追いかけ注意"
            plan["extra_warning"] = "反発足安値のブレイクはすでに発生済み。次の戻りまで待つ。"
        return plan

    # 最新の反発足がまだブレイクしていない場合は準備ラインを表示
    waiting = build_waiting_plan_at(df, i)
    if waiting is not None:
        return waiting

    row = df.iloc[i]
    reasons = get_range_exclusion_reasons(row)
    if reasons:
        detail = "レンジ除外：" + " / ".join(reasons)
    elif get_trade_direction(row) is None:
        detail = "15分足25MAの方向が明確でないため見送り"
    else:
        detail = "反発足とその高値/安値ブレイク、または加点条件1つ以上を待機"
    return {"valid": False, "status": "見送り", "detail": detail}


# ------------------------------------------------------------------
# 資金管理・バックテスト
# ------------------------------------------------------------------
def calc_position_size(capital: float, plan: dict) -> dict:
    allowed_loss = int(capital * RISK_PER_TRADE_PCT)
    if not plan.get("valid"):
        return {"allowed_loss": allowed_loss, "contracts": 0, "risk_per_contract": None}
    risk_per_contract = int(plan["risk"] * YEN_PER_POINT_PER_MICRO)
    contracts = int(allowed_loss // risk_per_contract) if risk_per_contract > 0 else 0
    return {"allowed_loss": allowed_loss, "risk_per_contract": risk_per_contract, "contracts": contracts}


def run_trade_outcome(df: pd.DataFrame, breakout_i: int, plan: dict) -> tuple[float | None, int | None]:
    """反発足高安ブレイクで約定した前提の簡易評価。

    同一バー内で利確・損切りが両方届く場合は、保守的に損切り優先で扱う。
    """
    direction = plan["direction"]
    entry = float(plan["entry"])
    stop = float(plan["stop"])
    target = float(plan["target1"])
    session_id = df.iloc[breakout_i]["SESSION_ID"]

    # ブレイクバー内で entry は成立済み。止まりと利確が同時なら保守的に損切り扱い。
    first = df.iloc[breakout_i]
    if direction == "BUY":
        if float(first["Low"]) <= stop:
            return -1.0, breakout_i
        if float(first["High"]) >= target:
            return TARGET_R1, breakout_i
    else:
        if float(first["High"]) >= stop:
            return -1.0, breakout_i
        if float(first["Low"]) <= target:
            return TARGET_R1, breakout_i

    last_i = min(len(df) - 1, breakout_i + BACKTEST_HOLD_BARS)
    for j in range(breakout_i + 1, last_i + 1):
        bar = df.iloc[j]
        if bar["SESSION_ID"] != session_id:
            break
        high = float(bar["High"])
        low = float(bar["Low"])
        if direction == "BUY":
            if low <= stop:
                return -1.0, j
            if high >= target:
                return TARGET_R1, j
        else:
            if high >= stop:
                return -1.0, j
            if low <= target:
                return TARGET_R1, j

    end_bar = df.iloc[min(last_i, len(df) - 1)]
    result_r = ((float(end_bar["Close"]) - entry) / plan["risk"]) if direction == "BUY" else ((entry - float(end_bar["Close"])) / plan["risk"])
    return float(np.clip(result_r, -1.0, TARGET_R1)), last_i


def backtest_strategy(df: pd.DataFrame, require_bonus: bool) -> pd.DataFrame:
    rows = []
    i = 45
    while i < len(df) - 1:
        plan = build_triggered_plan_at(df, i, require_bonus=require_bonus)
        if plan is None or not plan.get("valid"):
            i += 1
            continue
        result_r, exit_i = run_trade_outcome(df, i, plan)
        if result_r is not None:
            rows.append({
                "日時": df.index[i],
                "方向": "買い" if plan["direction"] == "BUY" else "売り",
                "結果R": result_r,
                "損切り幅": plan["risk"],
                "発動価格": plan["entry"],
                "加点数": plan["bonus_score"],
                "加点内容": " / ".join(plan["bonus_reasons"]),
                "損切り根拠": plan.get("stop_basis_label", ""),
            })
            i = max(i + BACKTEST_COOLDOWN_BARS, (exit_i or i) + 1)
        else:
            i += 1
    return pd.DataFrame(rows)


def summarize_backtest(trades: pd.DataFrame) -> dict | None:
    if trades.empty:
        return None
    r = trades["結果R"].astype(float)
    wins = r > 0
    streak = 0
    max_streak = 0
    for is_loss in (r <= 0).tolist():
        streak = streak + 1 if is_loss else 0
        max_streak = max(max_streak, streak)
    equity = r.cumsum()
    drawdown = equity - equity.cummax()
    losses = abs(r[r < 0].sum())
    return {
        "trades": len(trades),
        "win_rate": wins.mean() * 100,
        "avg_r": r.mean(),
        "profit_factor": r[r > 0].sum() / losses if losses > 0 else np.nan,
        "max_losing_streak": max_streak,
        "max_drawdown_r": drawdown.min(),
    }


def daily_reasons(row: pd.Series) -> list[str]:
    return [
        "終値が25日線より上" if row["Close"] > row["SMA25"] else "終値が25日線より下",
        "25日線が75日線より上" if row["SMA25"] > row["SMA75"] else "25日線が75日線より下",
        "日足MACDが上向き" if row["MACD"] > row["MACD_SIGNAL"] else "日足MACDが下向き",
    ]


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title(f"日経225先物｜反発足・ブレイク順張り {APP_VERSION}")
st.caption("必須4条件を満たし、加点条件が1つ以上ある時だけ採用する。損切りは反発シナリオ否定位置に置く。")

with st.sidebar:
    st.header("資金管理")
    capital = st.number_input(
        "口座資金（円）",
        min_value=10_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
    )
    st.metric("1回の許容損失（資金の1%）", f"{int(capital * RISK_PER_TRADE_PCT):,}円")
    st.caption("入力は口座資金のみ。枚数は構造的損切り幅から自動計算します。")
    if st.button("自動データを更新（日足＋5分足）", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.info(
    "参照価格は無料のNIY=F（CME日経円建て先物）です。日経225マイクロとは完全一致しません。"
    "発注前にマネックス側で同じ反発足高値・安値と損切り根拠を必ず確認してください。"
)

try:
    with st.spinner("日足と5分足を自動取得して判定しています..."):
        daily = add_daily_indicators(load_daily())
        intraday = add_intraday_indicators(load_intraday("5d"))
except Exception as error:
    st.error(f"データ取得エラー：{error}")
    st.stop()

current_i = get_last_completed_index(intraday)
current_row = intraday.iloc[current_i]
current_price = float(current_row["Close"])
last_bar_time = intraday.index[current_i]
minutes_old = max(0, int((pd.Timestamp.now(tz=JST) - last_bar_time).total_seconds() // 60))
latest_daily = daily.iloc[-1]
current_exclusion_reasons = get_range_exclusion_reasons(current_row)
current_plan = detect_current_setup(intraday, current_price)
position = calc_position_size(float(capital), current_plan)

range_window = intraday.iloc[max(0, current_i - RANGE_REFERENCE_BARS + 1) : current_i + 1]
range_high_9h = ceil_tick(range_window["High"].max())
range_low_9h = floor_tick(range_window["Low"].min())

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "総合判定", "条件チェック", "実行プラン", "簡易検証", "運用ルール"
])

with tab1:
    st.subheader("今の環境")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("日足の週方向", latest_daily["WEEK_SIGNAL"])
    c2.metric("15分足25MA", current_row["TREND15"])
    c3.metric("5分足終値", f"{current_price:,.0f}")
    c4.metric("5分足の最終時刻", last_bar_time.strftime("%m/%d %H:%M"))
    c5.metric("レンジ除外", "発動" if current_exclusion_reasons else "通過")

    if minutes_old > 20:
        st.warning(f"5分足が約{minutes_old}分前で止まっています。休場・遅延時は新規判断をしません。")

    if current_exclusion_reasons:
        st.error("見送り：" + " / ".join(current_exclusion_reasons))
    else:
        st.success("レンジ除外を通過。15分足方向と同じ側だけ、反発足・ブレイクを待ちます。")

    if current_plan.get("valid"):
        label = "買い" if current_plan["direction"] == "BUY" else "売り"
        if current_plan["status"] == "追いかけ注意":
            st.warning(f"総合判定：{label}・追いかけ注意")
            st.write(current_plan.get("extra_warning", ""))
        elif current_plan.get("triggered"):
            (st.success if current_plan["direction"] == "BUY" else st.error)(f"総合判定：{label}発動｜加点 {current_plan['bonus_score']}/4")
        else:
            (st.success if current_plan["direction"] == "BUY" else st.error)(f"総合判定：{label}準備｜反発足高安ブレイク待ち")
    else:
        st.warning("総合判定：見送り｜必須4条件と、加点条件1つ以上が揃うまで待機")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("日足の背景")
        for reason in daily_reasons(latest_daily):
            st.write(f"・{reason}")
    with right:
        st.subheader("現在のフィルター")
        st.write(f"・セッション：{current_row['SESSION_KIND']}")
        st.write(f"・EMA9 / EMA25：{current_row['EMA9_5']:,.0f} / {current_row['EMA25_5']:,.0f}")
        st.write(f"・約9時間レンジ：{range_low_9h:,} 〜 {range_high_9h:,}")
        st.write(f"・15分足25MAの90分変化：{abs(float(current_row['EMA25_CHANGE_N'])):.0f}円")
        st.write(f"・15分足MA跨ぎ：{int(round(float(current_row['SIDE_FLIP_COUNT_15'])))}回 / 直近約2時間")
        st.write(f"・5分足MA接触：{int(round(float(current_row['TOUCH_BAR_COUNT_5'])))}本、イベント{int(round(float(current_row['TOUCH_EVENT_COUNT_5'])))}回 / 直近約1時間")

    st.subheader("直近5分足")
    chart = intraday[["Close", "EMA9_5", "EMA25_5"]].tail(180).copy()
    if intraday["VWAP"].notna().any():
        chart["VWAP"] = intraday["VWAP"].tail(180)
    st.line_chart(chart)

with tab2:
    st.subheader("エントリー条件チェック")
    st.markdown("### 必須条件（すべて必要）")
    required_rows = [
        ["1. レンジ除外", "通過" if not current_exclusion_reasons else "未達", " / ".join(current_exclusion_reasons) if current_exclusion_reasons else "15分足横ばい・MA跨ぎ・5分足MA多重接触を回避"],
        ["2. 15分足25MAと同方向", "通過" if get_trade_direction(current_row) else "未達", str(current_row["TREND15"])],
        ["3. 5分足反発足＋高安ブレイク", "判定中", current_plan.get("setup", current_plan.get("detail", "反発足とブレイクを待機"))],
        ["4. 構造的損切り", "準備済み" if current_plan.get("valid") else "反発足完成後に算出", current_plan.get("stop_reason", "根拠足安値/高値と直近押し安値/戻り高値の外側")],
    ]
    st.dataframe(pd.DataFrame(required_rows, columns=["必須条件", "状態", "内容"]), hide_index=True, use_container_width=True)

    st.markdown("### 加点条件（1つ以上で採用）")
    if current_plan.get("bonus_items"):
        bonus_df = pd.DataFrame([
            {"加点条件": x["name"], "判定": "＋1" if x["passed"] else "加点なし", "内容": x["detail"]}
            for x in current_plan["bonus_items"]
        ])
        st.dataframe(bonus_df, hide_index=True, use_container_width=True)
        st.write(f"**加点数：{current_plan.get('bonus_score', 0)} / 4（必要：1以上）**")
    else:
        st.info("反発足が完成すると、加点条件を自動採点します。")

with tab3:
    st.subheader("実行プラン")
    if not current_plan.get("valid"):
        st.warning(current_plan.get("detail", "現在は見送りです。"))
        st.write("反発足・高安ブレイクが出た後も、加点条件が1つ以上ある時だけ採用します。")
    else:
        direction_label = "買い" if current_plan["direction"] == "BUY" else "売り"
        if current_plan["status"] == "追いかけ注意":
            st.warning(f"{direction_label}：追いかけ注意")
            st.write(current_plan.get("extra_warning", ""))
        elif current_plan.get("triggered"):
            (st.success if current_plan["direction"] == "BUY" else st.error)(f"{direction_label}発動：{current_plan['setup']}")
        else:
            (st.success if current_plan["direction"] == "BUY" else st.error)(f"{direction_label}準備：{current_plan['setup']}")
            st.caption("まだ未発動です。反発足の高値/安値を5円抜けた時だけエントリー条件が完成します。")

        a, b, c, d = st.columns(4)
        a.metric("発動ライン", f"{current_plan['entry']:,}")
        b.metric("損切り", f"{current_plan['stop']:,}")
        c.metric("第一利確 1.5R", f"{current_plan['target1']:,}")
        d.metric("第二利確 2R", f"{current_plan['target2']:,}")

        r1, r2, r3 = st.columns(3)
        r1.metric("損切り幅", f"{current_plan['risk']}円")
        r2.metric("1枚あたり最大損失", f"{position['risk_per_contract']:,}円")
        r3.metric("最大枚数", f"{position['contracts']}枚")

        st.info("損切り根拠：" + current_plan["stop_reason"])
        if current_plan.get("barrier") is not None:
            label = "抵抗帯" if current_plan["direction"] == "BUY" else "支持帯"
            st.caption(f"次の{label}（簡易判定）：{current_plan['barrier']:,.0f}円 / 到達余地 {current_plan['room_r']:.2f}R")
        elif np.isinf(current_plan.get("room_r", np.nan)):
            st.caption("直近約9時間には、1.5R未満の明確な逆方向スイングを検出していません。")

        st.markdown("#### 発注前チェック")
        for line in current_plan.get("mandatory_reasons", []):
            st.write(f"・必須：{line}")
        for line in current_plan.get("bonus_reasons", []):
            st.write(f"・加点：{line}")
        if position["contracts"] < 1:
            st.error("資金1%では1枚も許容できません。エントリーを見送るか、損切り構造が近い別セットアップを待ちます。")

with tab4:
    st.subheader("過去5日分の簡易検証")
    st.caption("無料5分足の範囲で、同一セッション内・第一利確1.5Rまたは構造的損切りを比較します。手数料・スリッページ・実際の約定順序は未反映です。")

    with st.spinner("必須条件のみ / 必須＋加点1つ以上 を比較中..."):
        trades_required = backtest_strategy(intraday, require_bonus=False)
        trades_final = backtest_strategy(intraday, require_bonus=True)

    compare = []
    for name, trades in [("必須条件のみ", trades_required), ("必須＋加点1つ以上", trades_final)]:
        summary = summarize_backtest(trades)
        if summary is None:
            compare.append({"条件": name, "件数": 0, "勝率": "-", "平均R": "-", "PF": "-", "最大連敗": "-", "最大DD(R)": "-"})
        else:
            compare.append({
                "条件": name,
                "件数": summary["trades"],
                "勝率": f"{summary['win_rate']:.1f}%",
                "平均R": f"{summary['avg_r']:+.2f}",
                "PF": f"{summary['profit_factor']:.2f}" if pd.notna(summary["profit_factor"]) else "-",
                "最大連敗": summary["max_losing_streak"],
                "最大DD(R)": f"{summary['max_drawdown_r']:.2f}",
            })
    st.dataframe(pd.DataFrame(compare), hide_index=True, use_container_width=True)

    st.markdown("#### 最終ルールで抽出された直近トレード")
    if trades_final.empty:
        st.info("直近5日では最終条件の該当がありません。条件が厳しいため、これは正常です。")
    else:
        show = trades_final.copy()
        show["日時"] = show["日時"].dt.strftime("%m/%d %H:%M")
        st.dataframe(show.tail(50), hide_index=True, use_container_width=True)

with tab5:
    st.subheader("このアプリの固定ルール")
    st.markdown("### 必須条件")
    st.write("1. 15分足25MAが横ばい・往復、または5分足25MAに何度も触れるレンジは除外")
    st.write("2. ロングは15分足25MAが明確に上向き、ショートは明確に下向きだけ")
    st.write("3. 5分足で反発足が出て、その高値/安値を5円抜けた時だけ発動")
    st.write("4. 損切りは固定幅でなく、根拠足と直近押し安値/戻り高値の外側")
    st.markdown("### 加点条件（1つ以上必要）")
    st.write("・5分足25MAへの接触が初回〜2回目")
    st.write("・ブレイク足に値幅・実体・終値位置の勢いがある")
    st.write("・直近約1時間の高値/安値を終値で明確に抜けている")
    st.write("・直近約9時間の次の抵抗帯/支持帯まで最低1.5Rある")
    st.warning("検証結果は、無料データかつ短期間の簡易検証です。実弾で使う前に、最低でも数十〜100件規模での検証と、マネックスの実際の価格での照合が必要です。")
