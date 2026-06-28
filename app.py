import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import time

# ============================================================
# 日経225先物｜順張り手法専用アプリ 戦略版 v3
# 参照データ: Yahoo Finance の NIY=F（CME Nikkei/Yen Futures）
# 発注前は必ずマネックスの日経225マイクロ価格・チャートで最終確認する。
# ============================================================

st.set_page_config(
    page_title="日経225先物｜順張り手法",
    page_icon="📈",
    layout="wide",
)

APP_VERSION = "戦略版 v3（明確トレンド・初回〜2回目・勢いブレイク）"
TICKER_INTRADAY = "NIY=F"
TICKER_DAILY = "^N225"
JST = "Asia/Tokyo"

# 日経225マイクロ向けの固定ルール
TICK = 5
YEN_PER_POINT_PER_MICRO = 10
RISK_PER_TRADE_PCT = 0.01
MAX_STOP_WIDTH = 100
STOP_BUFFER = 5
TARGET_R1 = 1.5
TARGET_R2 = 2.0

# 手法の固定パラメータ
OPENING_RANGE_BARS = 6             # 5分足6本＝寄り後30分
OPENING_VALID_BARS = 24            # 寄り後ブレイクを狙うのは最初の約2時間まで
PULLBACK_TOUCH_BARS = 4            # 直近4本以内に25EMAまで押す／戻す
PULLBACK_STOP_BARS = 5             # 損切りに使う直近の構造高安
RANGE_REFERENCE_BARS = 108         # 約9時間の重要レンジ確認用
BACKTEST_HOLD_BARS = 36            # 約3時間で決着しなければ手仕舞い評価
BACKTEST_COOLDOWN_BARS = 6

# レンジ・往復相場を避ける除外フィルター
TREND15_SLOPE_LOOKBACK = 6         # 15分足6本＝約90分
TREND15_FLAT_SLOPE_ATR = 0.35      # 90分のEMA変化が15分ATRの0.35倍未満なら横ばい
TREND15_CLEAR_SLOPE_ATR = 0.55     # 0.55倍以上でないと「明確な向き」と認めない
TREND15_CROSS_LOOKBACK = 8         # 15分足8本＝約2時間
TREND15_MAX_SIDE_FLIPS = 1         # 2回以上、終値がEMAを上下に跨げば除外
FIVE_MA_TOUCH_LOOKBACK = 12        # 5分足12本＝約1時間
FIVE_MA_MAX_TOUCHES = 3            # 4回以上触れていれば除外
MAX_VALID_TOUCH_EVENT_NO = 2       # 初回〜2回目の反発だけ採用
MA_TOUCH_TOLERANCE = TICK          # 25EMA±5円を「接触」とみなす

# ブレイクの勢い判定
BREAKOUT_MIN_RANGE_ATR = 0.70      # ブレイク足の値幅が5分ATRの0.7倍未満なら小さすぎる
BREAKOUT_MIN_BODY_RATIO = 0.55     # 実体が足全体の55%以上
BREAKOUT_MAX_WICK_RATIO = 0.25     # 逆方向ヒゲが足全体の25%以下
BREAKOUT_CLOSE_POSITION = 0.70     # 買いは高値圏、売りは安値圏で引ける必要


# ------------------------
# データ取得・整形
# ------------------------
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
        raise RuntimeError(
            "5分足を取得できませんでした。市場休場・Yahoo側の一時的な不具合・ティッカー更新の可能性があります。"
        )

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Volume"] = df["Volume"].fillna(0).clip(lower=0)

    if df.empty:
        raise RuntimeError("利用できる5分足がありません。")

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


# ------------------------
# 指標・セッション判定
# ------------------------
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
    bars15["EMA25_SLOPE_3"] = bars15["EMA25_15"] - bars15["EMA25_15"].shift(3)
    bars15["CLOSE_CHANGE_3"] = bars15["Close"] - bars15["Close"].shift(3)

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

    # 15分足25EMAの90分変化
    bars15["EMA25_CHANGE_N"] = bars15["EMA25_15"] - bars15["EMA25_15"].shift(TREND15_SLOPE_LOOKBACK)
    bars15["CLOSE_CHANGE_N"] = bars15["Close"] - bars15["Close"].shift(TREND15_SLOPE_LOOKBACK)
    bars15["EMA25_FLAT_15"] = bars15["EMA25_CHANGE_N"].abs() < bars15["ATR14_15"] * TREND15_FLAT_SLOPE_ATR
    bars15["EMA25_CLEAR_SLOPE_15"] = bars15["EMA25_CHANGE_N"].abs() >= bars15["ATR14_15"] * TREND15_CLEAR_SLOPE_ATR

    # 15分足の終値が25EMAを何度も跨いでいないか
    close_side_raw = pd.Series(
        np.select(
            [
                bars15["Close"] > bars15["EMA25_15"] + MA_TOUCH_TOLERANCE,
                bars15["Close"] < bars15["EMA25_15"] - MA_TOUCH_TOLERANCE,
            ],
            [1, -1],
            default=0,
        ),
        index=bars15.index,
    )
    bars15["CLOSE_SIDE_15"] = close_side_raw.replace(0, np.nan).ffill().fillna(0)
    prev_side = bars15["CLOSE_SIDE_15"].shift(1)
    bars15["SIDE_FLIP_15"] = (
        (bars15["CLOSE_SIDE_15"] != prev_side)
        & (bars15["CLOSE_SIDE_15"] != 0)
        & (prev_side != 0)
    ).astype(int)
    bars15["SIDE_FLIP_COUNT_15"] = bars15["SIDE_FLIP_15"].rolling(
        TREND15_CROSS_LOOKBACK,
        min_periods=TREND15_CROSS_LOOKBACK,
    ).sum()

    # 旧条件相当の方向（比較検証用）
    bars15["TREND15_RAW"] = np.select(
        [
            (bars15["Close"] > bars15["EMA25_15"])
            & (bars15["EMA25_SLOPE_3"] > 0)
            & (bars15["CLOSE_CHANGE_3"] > 0),
            (bars15["Close"] < bars15["EMA25_15"])
            & (bars15["EMA25_SLOPE_3"] < 0)
            & (bars15["CLOSE_CHANGE_3"] < 0),
        ],
        ["上昇", "下降"],
        default="レンジ",
    )

    # 新条件：15分足25EMAの「明確な」方向だけを採用する
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
    bars15["TREND15_STRICT_EXCLUDED"] = bars15["TREND15"] == "不明確"
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

    # 5分足25EMAへの接触。連続して触れている複数本は「1回の反発」と数える。
    out["TOUCH_5"] = (
        (out["Low"] <= out["EMA25_5"] + MA_TOUCH_TOLERANCE)
        & (out["High"] >= out["EMA25_5"] - MA_TOUCH_TOLERANCE)
    )
    out["TOUCH_EVENT_5"] = out.groupby("SESSION_ID")["TOUCH_5"].transform(
        lambda s: s.astype(bool) & ~s.astype(bool).shift(1, fill_value=False)
    )
    out["TOUCH_EVENT_NO_5"] = out.groupby("SESSION_ID")["TOUCH_EVENT_5"].cumsum().astype(int)
    out["TOUCH_COUNT_5"] = out.groupby("SESSION_ID")["TOUCH_5"].transform(
        lambda s: s.astype(int).rolling(FIVE_MA_TOUCH_LOOKBACK, min_periods=1).sum()
    )
    out["FIVE_MIN_CHOP_EXCLUDED"] = out["TOUCH_COUNT_5"] > FIVE_MA_MAX_TOUCHES
    out["FIVE_TOUCH_EXHAUSTED"] = out["TOUCH_EVENT_NO_5"] > MAX_VALID_TOUCH_EVENT_NO

    bars15 = build_15m_trend(out)
    out = pd.merge_asof(
        out.sort_index(),
        bars15[
            [
                "EMA25_15",
                "EMA25_SLOPE_3",
                "ATR14_15",
                "EMA25_CHANGE_N",
                "EMA25_FLAT_15",
                "EMA25_CLEAR_SLOPE_15",
                "SIDE_FLIP_COUNT_15",
                "TREND15_CHOP_EXCLUDED",
                "TREND15_STRICT_EXCLUDED",
                "TREND15_RAW",
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
            "EMA25_5",
            "MACD5_SIGNAL",
            "ATR14_5",
            "EMA25_15",
            "ATR14_15",
            "SIDE_FLIP_COUNT_15",
            "TOUCH_COUNT_5",
            "TREND15",
        ]
    ).copy()


# ------------------------
# フィルター・セットアップ共通
# ------------------------
def get_last_completed_index(df: pd.DataFrame) -> int:
    return max(0, len(df) - 2)


def get_range_exclusion_reasons(row: pd.Series) -> list[str]:
    reasons = []
    if bool(row.get("EMA25_FLAT_15", False)):
        change = abs(float(row.get("EMA25_CHANGE_N", 0.0)))
        atr = float(row.get("ATR14_15", 0.0))
        reasons.append(f"15分足25EMAが横ばい（90分の変化 {change:.0f}円、15分ATR {atr:.0f}円）")

    side_flips = int(round(float(row.get("SIDE_FLIP_COUNT_15", 0.0))))
    if side_flips > TREND15_MAX_SIDE_FLIPS:
        reasons.append(f"15分足終値が25EMAを直近約2時間で{side_flips}回跨いでいる")

    touches_5 = int(round(float(row.get("TOUCH_COUNT_5", 0.0))))
    if touches_5 > FIVE_MA_MAX_TOUCHES:
        reasons.append(f"5分足が25EMAに直近約1時間で{touches_5}回触れている")
    return reasons


def get_strict_exclusion_reasons(row: pd.Series, touch_event_no: int | None = None) -> list[str]:
    reasons = []
    trend = str(row.get("TREND15", "不明確"))
    if trend == "不明確":
        slope = abs(float(row.get("EMA25_CHANGE_N", 0.0)))
        atr = float(row.get("ATR14_15", 0.0))
        reasons.append(
            f"15分足25EMAの傾きが不十分（90分の変化 {slope:.0f}円、明確判定の目安 {atr * TREND15_CLEAR_SLOPE_ATR:.0f}円）"
        )

    event_no = int(row.get("TOUCH_EVENT_NO_5", 0)) if touch_event_no is None else int(touch_event_no)
    if event_no > MAX_VALID_TOUCH_EVENT_NO:
        reasons.append(f"5分足25EMAからの反発が{event_no}回目で、初回〜2回目の条件外")
    return reasons


def get_exclusion_reasons(
    row: pd.Series,
    apply_range_filter: bool = True,
    apply_strict_filter: bool = True,
    touch_event_no: int | None = None,
) -> list[str]:
    reasons = []
    if apply_range_filter:
        reasons.extend(get_range_exclusion_reasons(row))
    if apply_strict_filter:
        reasons.extend(get_strict_exclusion_reasons(row, touch_event_no=touch_event_no))
    return reasons


def get_trade_direction(row: pd.Series, strict: bool = True) -> str | None:
    if strict:
        if row["TREND15"] == "明確上昇":
            return "BUY"
        if row["TREND15"] == "明確下降":
            return "SELL"
        return None

    if row["TREND15_RAW"] == "上昇":
        return "BUY"
    if row["TREND15_RAW"] == "下降":
        return "SELL"
    return None


def latest_touch_event_no(df: pd.DataFrame, end_i: int) -> int:
    start_i = max(0, end_i - PULLBACK_TOUCH_BARS + 1)
    recent = df.iloc[start_i : end_i + 1]
    touched = recent[recent["TOUCH_5"]]
    if touched.empty:
        return 0
    return int(touched["TOUCH_EVENT_NO_5"].iloc[-1])


def candle_momentum_reasons(bar: pd.Series, direction: str, level: float) -> list[str]:
    """ブレイク足が小さい／長い逆ヒゲ／終値が伸びない状態を見送る。"""
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

    reasons = []
    if bar_range < atr * BREAKOUT_MIN_RANGE_ATR:
        reasons.append(f"ブレイク足の値幅が小さい（{bar_range:.0f}円、5分ATRの{bar_range / atr:.2f}倍）")
    if body_ratio < BREAKOUT_MIN_BODY_RATIO:
        reasons.append(f"ブレイク足の実体が小さい（実体比 {body_ratio:.0%}）")

    if direction == "BUY":
        if close <= opn:
            reasons.append("買いブレイク足が陽線ではない")
        if close < level + TICK:
            reasons.append("買いブレイク足の終値が直近高値の上に定着していない")
        if upper_wick_ratio > BREAKOUT_MAX_WICK_RATIO:
            reasons.append(f"買いブレイク足の上ヒゲが長い（{upper_wick_ratio:.0%}）")
        if close_position < BREAKOUT_CLOSE_POSITION:
            reasons.append("買いブレイク足が高値圏で引けていない")
    else:
        if close >= opn:
            reasons.append("売りブレイク足が陰線ではない")
        if close > level - TICK:
            reasons.append("売りブレイク足の終値が直近安値の下に定着していない")
        if lower_wick_ratio > BREAKOUT_MAX_WICK_RATIO:
            reasons.append(f"売りブレイク足の下ヒゲが長い（{lower_wick_ratio:.0%}）")
        if close_position > 1 - BREAKOUT_CLOSE_POSITION:
            reasons.append("売りブレイク足が安値圏で引けていない")
    return reasons


def confirmation_holds_ema(confirm_bar: pd.Series, direction: str, breakout_level: float) -> tuple[bool, str | None]:
    """ブレイク直後に25EMAへ戻ったら見送る。"""
    ema = float(confirm_bar["EMA25_5"])
    if direction == "BUY":
        if float(confirm_bar["Low"]) <= ema + MA_TOUCH_TOLERANCE:
            return False, "ブレイク直後の足が5分足25EMAへ戻っている"
        if float(confirm_bar["Close"]) < breakout_level:
            return False, "ブレイク直後の足が直近高値を維持できていない"
    else:
        if float(confirm_bar["High"]) >= ema - MA_TOUCH_TOLERANCE:
            return False, "ブレイク直後の足が5分足25EMAへ戻っている"
        if float(confirm_bar["Close"]) > breakout_level:
            return False, "ブレイク直後の足が直近安値を維持できていない"
    return True, None


def calculate_plan(direction: str, entry: float, stop: float, setup: str, reasons: list[str]) -> dict:
    entry = ceil_tick(entry) if direction == "BUY" else floor_tick(entry)
    stop = floor_tick(stop) if direction == "BUY" else ceil_tick(stop)
    risk = (entry - stop) if direction == "BUY" else (stop - entry)

    if risk <= 0:
        return {"valid": False, "reason": "損切り位置が不正です。"}
    if risk > MAX_STOP_WIDTH:
        return {
            "valid": False,
            "reason": f"必要な損切り幅が{risk:.0f}円で、上限{MAX_STOP_WIDTH}円を超えています。",
            "risk": risk,
        }

    target1 = ceil_tick(entry + risk * TARGET_R1) if direction == "BUY" else floor_tick(entry - risk * TARGET_R1)
    target2 = ceil_tick(entry + risk * TARGET_R2) if direction == "BUY" else floor_tick(entry - risk * TARGET_R2)
    return {
        "valid": True,
        "direction": direction,
        "entry": int(entry),
        "stop": int(stop),
        "risk": int(risk),
        "target1": int(target1),
        "target2": int(target2),
        "setup": setup,
        "reasons": reasons,
    }


# ------------------------
# 現在・過去共通のセットアップ判定
# ------------------------
def pullback_setup_at(
    df: pd.DataFrame,
    i: int,
    apply_range_filter: bool = True,
    apply_strict_filter: bool = True,
    apply_momentum_filter: bool = True,
) -> dict | None:
    """15分足トレンド＋5分足25EMA押し目／戻り売り。

    勢いフィルターON時は、
    反転足 → 勢いのあるブレイク足 → 25EMAへ戻らない確認足
    の3本を確認後、確認足の高値・安値を抜けたら発動とする。
    """
    min_required = max(40, PULLBACK_STOP_BARS + 4)
    if i < min_required:
        return None

    if apply_momentum_filter:
        reversal_i = i - 2
        breakout_i = i - 1
        confirm_i = i
    else:
        reversal_i = i
        breakout_i = None
        confirm_i = None

    reversal = df.iloc[reversal_i]
    touch_no = latest_touch_event_no(df, reversal_i)
    exclusion_reasons = get_exclusion_reasons(
        reversal,
        apply_range_filter=apply_range_filter,
        apply_strict_filter=apply_strict_filter,
        touch_event_no=touch_no,
    )
    if exclusion_reasons:
        return None

    direction = get_trade_direction(reversal, strict=apply_strict_filter)
    if direction is None:
        return None

    prev = df.iloc[reversal_i - 1]
    recent_start = max(0, reversal_i - PULLBACK_TOUCH_BARS + 1)
    recent = df.iloc[recent_start : reversal_i + 1]
    stop_start = max(0, reversal_i - PULLBACK_STOP_BARS + 1)
    stop_window = df.iloc[stop_start : reversal_i + 1]

    if direction == "BUY":
        touch = (recent["Low"] <= recent["EMA25_5"] + TICK).any()
        reversal_ok = (
            reversal["Close"] > reversal["Open"]
            and reversal["Close"] > prev["Close"]
            and reversal["Close"] > reversal["EMA25_5"]
        )
        if not (touch and reversal_ok):
            return None

        reasons = [
            "15分足25EMAが明確に上向き",
            f"5分足25EMAへの反発は{touch_no}回目（初回〜2回目）",
            "5分足が25EMA付近まで押して反転陽線を形成",
        ]
        if reversal["MACD5"] > reversal["MACD5_SIGNAL"]:
            reasons.append("5分足MACDも上向き")

        if not apply_momentum_filter:
            plan = calculate_plan(
                "BUY",
                reversal["High"] + TICK,
                stop_window["Low"].min() - STOP_BUFFER,
                "15分足上昇＋5分足25EMA押し目",
                reasons,
            )
            if plan.get("valid"):
                plan.update({"bar_time": reversal.name, "touch_event_no": touch_no})
            return plan

        breakout = df.iloc[breakout_i]
        confirm = df.iloc[confirm_i]
        level = float(reversal["High"])
        if float(breakout["High"]) < level + TICK:
            return None
        momentum_reasons = candle_momentum_reasons(breakout, "BUY", level)
        if momentum_reasons:
            return None
        holds, hold_reason = confirmation_holds_ema(confirm, "BUY", level)
        if not holds:
            return None
        reasons.extend([
            "直近高値ブレイク足に十分な実体・値幅がある",
            "ブレイク直後の足が5分足25EMAへ戻っていない",
        ])
        plan = calculate_plan(
            "BUY",
            max(float(breakout["High"]), float(confirm["High"])) + TICK,
            stop_window["Low"].min() - STOP_BUFFER,
            "明確上昇＋25EMA初回〜2回目押し目＋勢いブレイク",
            reasons,
        )
        if plan.get("valid"):
            plan.update({"bar_time": confirm.name, "touch_event_no": touch_no})
        return plan

    # SELL
    touch = (recent["High"] >= recent["EMA25_5"] - TICK).any()
    reversal_ok = (
        reversal["Close"] < reversal["Open"]
        and reversal["Close"] < prev["Close"]
        and reversal["Close"] < reversal["EMA25_5"]
    )
    if not (touch and reversal_ok):
        return None

    reasons = [
        "15分足25EMAが明確に下向き",
        f"5分足25EMAへの反発は{touch_no}回目（初回〜2回目）",
        "5分足が25EMA付近まで戻して反転陰線を形成",
    ]
    if reversal["MACD5"] < reversal["MACD5_SIGNAL"]:
        reasons.append("5分足MACDも下向き")

    if not apply_momentum_filter:
        plan = calculate_plan(
            "SELL",
            reversal["Low"] - TICK,
            stop_window["High"].max() + STOP_BUFFER,
            "15分足下降＋5分足25EMA戻り売り",
            reasons,
        )
        if plan.get("valid"):
            plan.update({"bar_time": reversal.name, "touch_event_no": touch_no})
        return plan

    breakout = df.iloc[breakout_i]
    confirm = df.iloc[confirm_i]
    level = float(reversal["Low"])
    if float(breakout["Low"]) > level - TICK:
        return None
    momentum_reasons = candle_momentum_reasons(breakout, "SELL", level)
    if momentum_reasons:
        return None
    holds, hold_reason = confirmation_holds_ema(confirm, "SELL", level)
    if not holds:
        return None
    reasons.extend([
        "直近安値ブレイク足に十分な実体・値幅がある",
        "ブレイク直後の足が5分足25EMAへ戻っていない",
    ])
    plan = calculate_plan(
        "SELL",
        min(float(breakout["Low"]), float(confirm["Low"])) - TICK,
        stop_window["High"].max() + STOP_BUFFER,
        "明確下降＋25EMA初回〜2回目戻り＋勢いブレイク",
        reasons,
    )
    if plan.get("valid"):
        plan.update({"bar_time": confirm.name, "touch_event_no": touch_no})
    return plan


def detect_pullback_setup(df: pd.DataFrame) -> dict:
    i = get_last_completed_index(df)
    plan = pullback_setup_at(
        df,
        i,
        apply_range_filter=True,
        apply_strict_filter=True,
        apply_momentum_filter=True,
    )
    if plan is not None:
        plan["status"] = "買い待機" if plan.get("direction") == "BUY" else "売り待機"
        if not plan.get("valid"):
            plan["status"] = "見送り"
        return plan

    row = df.iloc[i]
    exclusion_reasons = get_exclusion_reasons(row, True, True)
    if exclusion_reasons:
        detail = "除外条件：" + " / ".join(exclusion_reasons)
    elif row["TREND15"] == "不明確":
        detail = "15分足25EMAの向きが明確ではないため、順張りをしません。"
    elif row["TREND15"] == "明確上昇":
        detail = "15分足は明確上昇ですが、初回〜2回目の25EMA押し目＋勢いブレイク条件が未完成です。"
    else:
        detail = "15分足は明確下降ですが、初回〜2回目の25EMA戻り＋勢いブレイク条件が未完成です。"
    return {"valid": False, "status": "見送り", "detail": detail, "bar_time": row.name}


def detect_opening_breakout_setup(df: pd.DataFrame) -> dict:
    i = get_last_completed_index(df)
    row = df.iloc[i]
    exclusion_reasons = get_exclusion_reasons(row, True, True)
    if exclusion_reasons:
        return {"valid": False, "status": "見送り", "detail": "除外条件：" + " / ".join(exclusion_reasons)}

    if row["SESSION_KIND"] == "時間外":
        return {"valid": False, "status": "対象外", "detail": "日中・夜間セッション外のため寄り後ブレイクは判定しません。"}

    session_id = row["SESSION_ID"]
    session = df[(df["SESSION_ID"] == session_id) & (df.index <= row.name)].copy()
    if len(session) < OPENING_RANGE_BARS + 3:
        return {"valid": False, "status": "待機", "detail": "寄り後30分レンジとブレイク確認足の形成待ちです。"}
    if len(session) > OPENING_VALID_BARS:
        return {"valid": False, "status": "時間切れ", "detail": "寄り後ブレイクを狙う時間帯を過ぎています。"}

    direction = get_trade_direction(row, strict=True)
    if direction is None:
        return {"valid": False, "status": "見送り", "detail": "15分足25EMAの方向が明確ではありません。"}

    opening = session.iloc[:OPENING_RANGE_BARS]
    range_high = ceil_tick(opening["High"].max())
    range_low = floor_tick(opening["Low"].min())
    breakout = session.iloc[-2]
    confirm = session.iloc[-1]
    recent_stop = session.iloc[-min(PULLBACK_STOP_BARS, len(session)) :]

    if direction == "BUY":
        if float(breakout["High"]) < range_high + TICK:
            return {"valid": False, "status": "待機", "detail": f"寄り後30分高値 {range_high:,}円の上抜け待ちです。", "range_high": range_high, "range_low": range_low}
        bad = candle_momentum_reasons(breakout, "BUY", range_high)
        if bad:
            return {"valid": False, "status": "見送り", "detail": "ブレイクの勢い不足：" + " / ".join(bad), "range_high": range_high, "range_low": range_low}
        holds, reason = confirmation_holds_ema(confirm, "BUY", range_high)
        if not holds:
            return {"valid": False, "status": "見送り", "detail": reason, "range_high": range_high, "range_low": range_low}
        plan = calculate_plan(
            "BUY",
            max(float(breakout["High"]), float(confirm["High"])) + TICK,
            recent_stop["Low"].min() - STOP_BUFFER,
            f"{row['SESSION_KIND']}寄り後30分レンジ・勢い上抜け",
            [
                f"寄り後30分高値 {range_high:,}円を強い陽線で上抜け",
                "ブレイク足の実体・値幅・ヒゲが基準を通過",
                "確認足が5分足25EMAへ戻っていない",
                "15分足25EMAが明確に上向き",
            ],
        )
    else:
        if float(breakout["Low"]) > range_low - TICK:
            return {"valid": False, "status": "待機", "detail": f"寄り後30分安値 {range_low:,}円の下抜け待ちです。", "range_high": range_high, "range_low": range_low}
        bad = candle_momentum_reasons(breakout, "SELL", range_low)
        if bad:
            return {"valid": False, "status": "見送り", "detail": "ブレイクの勢い不足：" + " / ".join(bad), "range_high": range_high, "range_low": range_low}
        holds, reason = confirmation_holds_ema(confirm, "SELL", range_low)
        if not holds:
            return {"valid": False, "status": "見送り", "detail": reason, "range_high": range_high, "range_low": range_low}
        plan = calculate_plan(
            "SELL",
            min(float(breakout["Low"]), float(confirm["Low"])) - TICK,
            recent_stop["High"].max() + STOP_BUFFER,
            f"{row['SESSION_KIND']}寄り後30分レンジ・勢い下抜け",
            [
                f"寄り後30分安値 {range_low:,}円を強い陰線で下抜け",
                "ブレイク足の実体・値幅・ヒゲが基準を通過",
                "確認足が5分足25EMAへ戻っていない",
                "15分足25EMAが明確に下向き",
            ],
        )

    plan["status"] = "買い待機" if plan.get("valid") and plan.get("direction") == "BUY" else "売り待機"
    if not plan.get("valid"):
        plan["status"] = "見送り"
    plan["range_high"] = range_high
    plan["range_low"] = range_low
    return plan


def choose_main_plan(pullback: dict, breakout: dict, current_price: float) -> dict:
    for candidate in [pullback, breakout]:
        if candidate.get("valid"):
            plan = candidate.copy()
            if plan["direction"] == "BUY" and current_price >= plan["entry"] + max(TICK, plan["risk"] * 0.5):
                plan["status"] = "追いかけ注意"
                plan["extra_warning"] = "発動ラインからすでに離れています。次の初回〜2回目押し目を待つ方が安全です。"
            elif plan["direction"] == "SELL" and current_price <= plan["entry"] - max(TICK, plan["risk"] * 0.5):
                plan["status"] = "追いかけ注意"
                plan["extra_warning"] = "発動ラインからすでに離れています。次の初回〜2回目戻りを待つ方が安全です。"
            return plan
    return {
        "valid": False,
        "status": "見送り",
        "reason": "明確な15分足トレンド・初回〜2回目のMA反発・勢いブレイクのいずれかが未達です。",
    }


def calc_position_size(capital: float, plan: dict) -> dict:
    allowed_loss = int(capital * RISK_PER_TRADE_PCT)
    if not plan.get("valid"):
        return {"allowed_loss": allowed_loss, "contracts": 0, "risk_per_contract": None}
    risk_per_contract = int(plan["risk"] * YEN_PER_POINT_PER_MICRO)
    contracts = int(allowed_loss // risk_per_contract) if risk_per_contract > 0 else 0
    return {"allowed_loss": allowed_loss, "risk_per_contract": risk_per_contract, "contracts": contracts}


def daily_reasons(row: pd.Series) -> list[str]:
    return [
        "終値が25日線より上" if row["Close"] > row["SMA25"] else "終値が25日線より下",
        "25日線が75日線より上" if row["SMA25"] > row["SMA75"] else "25日線が75日線より下",
        "日足MACDが上向き" if row["MACD"] > row["MACD_SIGNAL"] else "日足MACDが下向き",
    ]


# ------------------------
# 過去検証（簡易）
# ------------------------
def run_trade_outcome(df: pd.DataFrame, start_i: int, plan: dict) -> tuple[float | None, int | None]:
    """発動後、TP1=1.5Rか損切りのどちらが先かを保守的に判定する。"""
    direction = plan["direction"]
    entry = plan["entry"]
    stop = plan["stop"]
    target = plan["target1"]
    session_id = df.iloc[start_i]["SESSION_ID"]
    entered = False

    last_i = min(len(df) - 1, start_i + BACKTEST_HOLD_BARS)
    for j in range(start_i + 1, last_i + 1):
        bar = df.iloc[j]
        if bar["SESSION_ID"] != session_id:
            break

        high = float(bar["High"])
        low = float(bar["Low"])

        if not entered:
            if direction == "BUY" and high >= entry:
                entered = True
                if low <= stop:
                    return -1.0, j
                if high >= target:
                    return TARGET_R1, j
            elif direction == "SELL" and low <= entry:
                entered = True
                if high >= stop:
                    return -1.0, j
                if low <= target:
                    return TARGET_R1, j
            continue

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

    if not entered:
        return None, None

    end_bar = df.iloc[min(last_i, len(df) - 1)]
    if direction == "BUY":
        result_r = (float(end_bar["Close"]) - entry) / plan["risk"]
    else:
        result_r = (entry - float(end_bar["Close"])) / plan["risk"]
    return float(np.clip(result_r, -1.0, TARGET_R1)), last_i


def backtest_pullback(
    df: pd.DataFrame,
    apply_range_filter: bool,
    apply_strict_filter: bool,
    apply_momentum_filter: bool,
) -> pd.DataFrame:
    rows = []
    i = 45
    while i < len(df) - 2:
        plan = pullback_setup_at(
            df,
            i,
            apply_range_filter=apply_range_filter,
            apply_strict_filter=apply_strict_filter,
            apply_momentum_filter=apply_momentum_filter,
        )
        if plan is None or not plan.get("valid"):
            i += 1
            continue

        result_r, exit_i = run_trade_outcome(df, i, plan)
        if result_r is not None:
            rows.append(
                {
                    "日時": df.index[i],
                    "方向": "買い" if plan["direction"] == "BUY" else "売り",
                    "結果R": result_r,
                    "損切り幅": plan["risk"],
                    "発動価格": plan["entry"],
                    "25EMA反発回数": plan.get("touch_event_no", np.nan),
                }
            )
            i = max(i + BACKTEST_COOLDOWN_BARS, (exit_i or i) + 1)
        else:
            i += 1
    return pd.DataFrame(rows)


def summarize_backtest(trades: pd.DataFrame) -> dict | None:
    if trades.empty:
        return None
    r = trades["結果R"].astype(float)
    wins = r > 0

    max_streak = 0
    current = 0
    for x in (r <= 0).tolist():
        if x:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    equity = r.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "trades": len(trades),
        "win_rate": wins.mean() * 100,
        "avg_r": r.mean(),
        "profit_factor": r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.nan,
        "max_losing_streak": max_streak,
        "max_drawdown_r": drawdown.min(),
    }


# ------------------------
# UI
# ------------------------
st.title(f"日経225先物｜順張り手法 {APP_VERSION}")
st.caption(
    "明確な15分足25EMAの方向だけに絞り、5分足25EMAの初回〜2回目反発と、勢いのある直近高安ブレイクだけを狙う判定アプリ"
)

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
    st.caption("入力する数値は口座資金だけです。許容損失は資金の1%で固定しています。")
    if st.button("自動データを更新（日足＋5分足）", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.info(
    "このアプリは NIY=F（CME日経円建て先物）の無料5分足を参照します。"
    "日経225マイクロと完全一致しないため、実際の発注ラインは必ずマネックスのチャートで同じ高値・安値を確認してください。"
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
latest_daily = daily.iloc[-1]

pullback = detect_pullback_setup(intraday)
breakout = detect_opening_breakout_setup(intraday)
main_plan = choose_main_plan(pullback, breakout, current_price)
position = calc_position_size(float(capital), main_plan)
current_exclusion_reasons = get_exclusion_reasons(current_row, True, True)

last_bar_time = intraday.index[current_i]
minutes_old = max(0, int((pd.Timestamp.now(tz=JST) - last_bar_time).total_seconds() // 60))

range_window = intraday.iloc[max(0, current_i - RANGE_REFERENCE_BARS + 1) : current_i + 1]
range_high_9h = ceil_tick(range_window["High"].max())
range_low_9h = floor_tick(range_window["Low"].min())


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["総合判定", "セットアップ", "実行プラン", "簡易検証", "運用ルール"]
)

with tab1:
    st.subheader("今はどちら側だけを狙うか")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("日足の週方向", latest_daily["WEEK_SIGNAL"])
    c2.metric("15分足25EMA", current_row["TREND15"])
    c3.metric("5分足終値", f"{current_price:,.0f}")
    c4.metric("5分足の最終時刻", last_bar_time.strftime("%m/%d %H:%M"))
    c5.metric("フィルター", "除外" if current_exclusion_reasons else "通過")

    if minutes_old > 20:
        st.warning(f"5分足が約{minutes_old}分前で止まっています。休場・データ遅延時は新規判断をしません。")

    if current_exclusion_reasons:
        st.error("見送りフィルターが発動：" + " / ".join(current_exclusion_reasons))
    else:
        st.success("レンジ・微妙なトレンド・MA多重反発のフィルターを通過。セットアップだけを待ちます。")

    if main_plan.get("valid"):
        if main_plan["direction"] == "BUY":
            st.success(f"総合判定：買い待機｜{main_plan['setup']}")
        else:
            st.error(f"総合判定：売り待機｜{main_plan['setup']}")
    else:
        st.warning("総合判定：見送り｜明確トレンド・初回〜2回目反発・勢いブレイクが揃うまで待機")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("日足の背景")
        for reason in daily_reasons(latest_daily):
            st.write(f"・{reason}")
    with right:
        st.subheader("現在の5分足・フィルター")
        st.write(f"・セッション：{current_row['SESSION_KIND']}")
        st.write(f"・EMA9 / EMA25：{current_row['EMA9_5']:,.0f} / {current_row['EMA25_5']:,.0f}")
        if pd.notna(current_row["VWAP"]):
            relation = "上" if current_price > current_row["VWAP"] else "下"
            st.write(f"・VWAP：{current_row['VWAP']:,.0f}（終値はVWAPより{relation}）")
        else:
            st.write("・VWAP：参照データの出来高不足で未判定")
        st.write(f"・約9時間レンジ：{range_low_9h:,} 〜 {range_high_9h:,}")
        st.write(
            f"・15分足25EMAの90分変化：{abs(float(current_row['EMA25_CHANGE_N'])):.0f}円 "
            f"（明確判定の目安 {float(current_row['ATR14_15']) * TREND15_CLEAR_SLOPE_ATR:.0f}円）"
        )
        st.write(
            f"・15分足25EMAの跨ぎ：{int(round(float(current_row['SIDE_FLIP_COUNT_15'])))}回 / 直近約2時間"
        )
        st.write(
            f"・5分足25EMAの連続接触：{int(round(float(current_row['TOUCH_COUNT_5'])))}本 / 直近約1時間"
        )
        st.write(
            f"・5分足25EMAへの反発回数：{int(current_row['TOUCH_EVENT_NO_5'])}回 / 現セッション（3回目以降は除外）"
        )

    st.subheader("直近の5分足チャート")
    chart = intraday[["Close", "EMA9_5", "EMA25_5"]].tail(180).copy()
    if "VWAP" in intraday.columns and intraday["VWAP"].notna().any():
        chart["VWAP"] = intraday["VWAP"].tail(180)
    st.line_chart(chart)

with tab2:
    st.subheader("手法ごとのセットアップ判定")

    if current_exclusion_reasons:
        st.error("先にフィルターが発動しているため、セットアップは採用しません。\n\n・" + "\n・".join(current_exclusion_reasons))
    else:
        st.success("明確な15分足トレンド・MA多重接触除外を通過しています。次にセットアップ条件を確認します。")

    left, right = st.columns(2)
    with left:
        st.markdown("### ① 15分足トレンド＋5分足25EMA押し目／戻り売り")
        if pullback.get("valid"):
            color = st.success if pullback["direction"] == "BUY" else st.error
            color(f"{pullback['status']}：{pullback['setup']}")
            for reason in pullback.get("reasons", []):
                st.write(f"・{reason}")
        else:
            st.warning(f"{pullback.get('status', '見送り')}：{pullback.get('detail', pullback.get('reason', '条件未達'))}")

    with right:
        st.markdown("### ② 寄り後30分レンジの方向ブレイク")
        if breakout.get("valid"):
            color = st.success if breakout["direction"] == "BUY" else st.error
            color(f"{breakout['status']}：{breakout['setup']}")
            for reason in breakout.get("reasons", []):
                st.write(f"・{reason}")
        else:
            st.warning(f"{breakout.get('status', '見送り')}：{breakout.get('detail', breakout.get('reason', '条件未達'))}")
            if "range_high" in breakout:
                st.write(f"・寄り後レンジ：{breakout['range_low']:,} 〜 {breakout['range_high']:,}")

    st.divider()
    st.caption(
        "勢いブレイクは、反転足 → 直近高安を強い実体で抜ける足 → 25EMAへ戻らない確認足、の順に確認します。"
        "小さいブレイク足・長い逆ヒゲ・直後のMA回帰は見送りします。"
    )

with tab3:
    st.subheader("今この瞬間の実行プラン")

    if not main_plan.get("valid"):
        st.warning(main_plan.get("reason", "現在は見送りです。"))
        st.write("次の順張りセットアップが完成するまで、エントリーしません。")
    else:
        direction_label = "買い" if main_plan["direction"] == "BUY" else "売り"
        if main_plan["status"] == "追いかけ注意":
            st.warning(f"{direction_label}：追いかけ注意")
            st.write(main_plan.get("extra_warning", ""))
        elif main_plan["direction"] == "BUY":
            st.success(f"{direction_label}待機：{main_plan['setup']}")
        else:
            st.error(f"{direction_label}待機：{main_plan['setup']}")

        a, b, c, d = st.columns(4)
        a.metric("発動ライン", f"{main_plan['entry']:,}")
        b.metric("損切り", f"{main_plan['stop']:,}")
        c.metric("第一利確 1.5R", f"{main_plan['target1']:,}")
        d.metric("第二利確 2R", f"{main_plan['target2']:,}")

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("損切り幅", f"{main_plan['risk']}円")
        r2.metric("1枚あたり最大損失", f"{position['risk_per_contract']:,}円")
        r3.metric("最大枚数", f"{position['contracts']}枚")

        if position["contracts"] < 1:
            st.error("資金1%ルールでは1枚も許容できません。損切りが狭い次のセットアップを待ちます。")
        else:
            st.caption(
                f"資金 {int(capital):,}円 × 1% ＝ 許容損失 {position['allowed_loss']:,}円。"
                "上限枚数ではなく、まずは1枚でルール通りに検証するのが安全です。"
            )

        st.subheader("発注前チェック")
        checks = [
            "マネックスの日経225マイクロでも、同じ5分足の高値・安値を確認した",
            "15分足25EMAが明確に同方向を向いている",
            "5分足25EMAへの反発は初回または2回目だけである",
            "ブレイク足の実体が大きく、逆方向のヒゲが長くない",
            "ブレイク直後に5分足25EMAへ戻っていない",
            "発動ライン到達前に成行で入らない",
            "損切り注文を同時に置ける",
        ]
        for check in checks:
            st.write(f"□ {check}")

with tab4:
    st.subheader("15分足トレンド＋5分足25EMA押し目／戻り売り｜簡易検証")
    st.caption(
        "直近30日程度の無料5分足で、利確1.5Rと損切り1Rのどちらが先かを保守的に集計します。"
        "3段階を比較して、取引回数を減らしてでも平均R・最大DD・最大連敗が改善するかを確認します。"
    )

    if st.button("過去30日を検証する", use_container_width=True):
        try:
            with st.spinner("過去30日分の5分足を取得し、条件別に比較しています..."):
                history = add_intraday_indicators(load_intraday("30d"))
                st.session_state["bt_base"] = backtest_pullback(
                    history,
                    apply_range_filter=False,
                    apply_strict_filter=False,
                    apply_momentum_filter=False,
                )
                st.session_state["bt_range"] = backtest_pullback(
                    history,
                    apply_range_filter=True,
                    apply_strict_filter=False,
                    apply_momentum_filter=False,
                )
                st.session_state["bt_strict"] = backtest_pullback(
                    history,
                    apply_range_filter=True,
                    apply_strict_filter=True,
                    apply_momentum_filter=True,
                )
        except Exception as error:
            st.error(f"検証データの取得に失敗しました：{error}")

    trade_sets = [
        ("基本（旧条件）", st.session_state.get("bt_base")),
        ("レンジ除外あり", st.session_state.get("bt_range")),
        ("新条件すべて", st.session_state.get("bt_strict")),
    ]
    available = [(name, df_) for name, df_ in trade_sets if isinstance(df_, pd.DataFrame)]

    if available:
        comparison = []
        for name, trades in available:
            summary = summarize_backtest(trades)
            if summary is not None:
                comparison.append(
                    {
                        "検証": name,
                        "トレード数": f"{summary['trades']}回",
                        "勝率": f"{summary['win_rate']:.1f}%",
                        "平均R": f"{summary['avg_r']:+.2f}R",
                        "PF": "-" if pd.isna(summary['profit_factor']) else f"{summary['profit_factor']:.2f}",
                        "最大連敗": f"{summary['max_losing_streak']}回",
                        "最大DD": f"{summary['max_drawdown_r']:.2f}R",
                    }
                )
        if comparison:
            st.subheader("条件別の比較")
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
            st.caption(
                "採用判断は勝率だけでなく、平均R・PF・最大DD・最大連敗が改善するかで行います。"
                "『新条件すべて』でトレード数が大きく減っても、平均RやDDが悪化するならフィルターは採用しません。"
            )

            strict_trades = st.session_state.get("bt_strict")
            if isinstance(strict_trades, pd.DataFrame) and not strict_trades.empty:
                st.subheader("新条件すべて｜採用されたトレード一覧")
                shown = strict_trades.copy()
                shown["日時"] = shown["日時"].dt.strftime("%m/%d %H:%M")
                shown["結果R"] = shown["結果R"].map(lambda x: f"{x:+.2f}R")
                shown["25EMA反発回数"] = shown["25EMA反発回数"].astype(int).astype(str) + "回目"
                st.dataframe(shown.tail(100), use_container_width=True, hide_index=True)
        else:
            st.warning("条件に合うトレードが見つかりませんでした。期間が短いか、条件が厳しすぎる可能性があります。")
    else:
        st.info("上の『過去30日を検証する』を押すと、基本・レンジ除外・新条件すべてを比較表示します。")

    st.warning(
        "この検証は無料データの簡易版です。手数料、スリッページ、日経225マイクロとの価格差、同一足内の約定順は厳密に反映していません。"
    )

with tab5:
    st.subheader("このアプリで守るルール")
    rules = [
        "15分足25EMAの90分傾きが十分に大きい時だけ、同方向を狙う。横ばい・微妙な傾きは完全に除外する。",
        "15分足終値が約2時間で25EMAを2回以上跨いだら、往復相場として除外する。",
        "5分足25EMAの連続接触が約1時間で4本以上なら、レンジとして除外する。",
        "5分足25EMAからの反発は、現セッションで初回〜2回目だけを採用する。3回目以降は見送る。",
        "買いは明確上昇＋押し目、売りは明確下降＋戻り売りだけを狙う。",
        "ブレイク足が小さい、実体が小さい、逆方向ヒゲが長い、またはブレイク直後にMAへ戻るなら見送る。",
        "損切りが100円を超える形は、期待値があっても見送る。",
        "1回の許容損失は資金の1%。連敗しても枚数を増やさない。",
        "アプリは候補を出すだけ。発注はマネックスの実際の価格・チャートで最終確認する。",
    ]
    for n, rule in enumerate(rules, start=1):
        st.write(f"{n}. {rule}")
