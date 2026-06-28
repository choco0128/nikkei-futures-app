import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import time

# ============================================================
# 日経225先物｜順張り手法専用アプリ 戦略版 v1
# 参照データ: Yahoo Finance の NIY=F（CME Nikkei/Yen Futures）
# 発注前は必ずマネックスの日経225マイクロ価格・チャートで最終確認する。
# ============================================================

st.set_page_config(
    page_title="日経225先物｜順張り手法",
    page_icon="📈",
    layout="wide",
)

APP_VERSION = "戦略版 v1"
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
    # 夜間17:00以降を翌取引日扱いに寄せる
    local = ts.tz_convert(JST) if ts.tzinfo else ts
    return (local + pd.Timedelta(hours=7)).date()


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SESSION_KIND"] = [get_session_kind(ts) for ts in out.index]
    out["TRADE_DATE"] = [get_trade_date(ts) for ts in out.index]
    out["SESSION_ID"] = (
        out["TRADE_DATE"].astype(str) + "_" + out["SESSION_KIND"].astype(str)
    )
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
    bars15["EMA25_SLOPE"] = bars15["EMA25_15"] - bars15["EMA25_15"].shift(3)
    bars15["CLOSE_CHANGE_3"] = bars15["Close"] - bars15["Close"].shift(3)

    bars15["TREND15"] = np.select(
        [
            (bars15["Close"] > bars15["EMA25_15"])
            & (bars15["EMA25_SLOPE"] > 0)
            & (bars15["CLOSE_CHANGE_3"] > 0),
            (bars15["Close"] < bars15["EMA25_15"])
            & (bars15["EMA25_SLOPE"] < 0)
            & (bars15["CLOSE_CHANGE_3"] < 0),
        ],
        ["上昇", "下降"],
        default="レンジ",
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

    # Yahoo側の出来高が利用できる場合だけVWAPを算出する
    if out["Volume"].sum() > 0:
        typical = (out["High"] + out["Low"] + out["Close"]) / 3
        pv = (typical * out["Volume"]).groupby(out["SESSION_ID"]).cumsum()
        vol = out["Volume"].groupby(out["SESSION_ID"]).cumsum()
        out["VWAP"] = pv / vol.replace(0, np.nan)
    else:
        out["VWAP"] = np.nan

    bars15 = build_15m_trend(out)
    out = pd.merge_asof(
        out.sort_index(),
        bars15[["EMA25_15", "EMA25_SLOPE", "TREND15"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    out.index.name = "Datetime"
    return out.dropna(subset=["EMA25_5", "MACD5_SIGNAL", "ATR14_5", "EMA25_15", "TREND15"]).copy()


# ------------------------
# 現在のセットアップ判定
# ------------------------
def get_last_completed_index(df: pd.DataFrame) -> int:
    # 未完成の可能性がある最後の足を避け、ひとつ前の5分足を使う
    return max(0, len(df) - 2)


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


def detect_pullback_setup(df: pd.DataFrame) -> dict:
    i = get_last_completed_index(df)
    min_required = max(30, PULLBACK_STOP_BARS + 2)
    if i < min_required:
        return {"status": "判定不足", "detail": "5分足データが不足しています。"}

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    recent = df.iloc[i - PULLBACK_TOUCH_BARS + 1 : i + 1]
    stop_window = df.iloc[i - PULLBACK_STOP_BARS + 1 : i + 1]

    # 買い：15分足上昇＋25EMAまで押し＋陽線で再上昇
    buy_touch = (recent["Low"] <= recent["EMA25_5"] + TICK).any()
    buy_reclaim = row["Close"] > row["EMA25_5"]
    buy_reversal = (row["Close"] > row["Open"]) and (row["Close"] > prev["Close"])
    buy_macd = row["MACD5"] > row["MACD5_SIGNAL"]

    # 売り：15分足下降＋25EMAまで戻し＋陰線で再下落
    sell_touch = (recent["High"] >= recent["EMA25_5"] - TICK).any()
    sell_reclaim = row["Close"] < row["EMA25_5"]
    sell_reversal = (row["Close"] < row["Open"]) and (row["Close"] < prev["Close"])
    sell_macd = row["MACD5"] < row["MACD5_SIGNAL"]

    if row["TREND15"] == "上昇" and buy_touch and buy_reclaim and buy_reversal:
        reasons = [
            "15分足が上昇トレンド",
            "5分足が25EMA付近まで押した",
            "反転陽線で5分足25EMAを回復",
        ]
        if buy_macd:
            reasons.append("5分足MACDも上向き")
        plan = calculate_plan(
            "BUY",
            row["High"] + TICK,
            stop_window["Low"].min() - STOP_BUFFER,
            "15分足上昇＋5分足25EMA押し目",
            reasons,
        )
        plan["status"] = "買い待機" if plan.get("valid") else "見送り"
        plan["bar_time"] = row.name
        return plan

    if row["TREND15"] == "下降" and sell_touch and sell_reclaim and sell_reversal:
        reasons = [
            "15分足が下降トレンド",
            "5分足が25EMA付近まで戻した",
            "反転陰線で5分足25EMAを下回った",
        ]
        if sell_macd:
            reasons.append("5分足MACDも下向き")
        plan = calculate_plan(
            "SELL",
            row["Low"] - TICK,
            stop_window["High"].max() + STOP_BUFFER,
            "15分足下降＋5分足25EMA戻り売り",
            reasons,
        )
        plan["status"] = "売り待機" if plan.get("valid") else "見送り"
        plan["bar_time"] = row.name
        return plan

    reasons = []
    if row["TREND15"] == "レンジ":
        reasons.append("15分足がレンジで、順張りの方向がありません")
    elif row["TREND15"] == "上昇":
        reasons.append("15分足は上昇だが、25EMA押し目からの反転条件が未完成です")
    else:
        reasons.append("15分足は下降だが、25EMA戻りからの反転条件が未完成です")

    return {
        "status": "見送り",
        "detail": " / ".join(reasons),
        "bar_time": row.name,
    }


def detect_opening_breakout_setup(df: pd.DataFrame) -> dict:
    i = get_last_completed_index(df)
    row = df.iloc[i]
    session_id = row["SESSION_ID"]
    session = df[df["SESSION_ID"] == session_id].copy()
    session = session[session.index <= row.name]

    if row["SESSION_KIND"] == "時間外":
        return {"status": "対象外", "detail": "日中・夜間セッション外のため寄り後ブレイクは判定しません。"}
    if len(session) < OPENING_RANGE_BARS:
        return {"status": "待機", "detail": "寄り後30分レンジの形成待ちです。"}
    if len(session) > OPENING_VALID_BARS:
        return {"status": "時間切れ", "detail": "寄り後ブレイクを狙う時間帯を過ぎています。"}

    opening = session.iloc[:OPENING_RANGE_BARS]
    range_high = ceil_tick(opening["High"].max())
    range_low = floor_tick(opening["Low"].min())
    recent_stop = session.iloc[-min(PULLBACK_STOP_BARS, len(session)) :]

    if row["TREND15"] == "上昇":
        plan = calculate_plan(
            "BUY",
            range_high + TICK,
            recent_stop["Low"].min() - STOP_BUFFER,
            f"{row['SESSION_KIND']}寄り後30分レンジ上抜け",
            [
                f"寄り後30分高値 {range_high:,}円を上抜け待ち",
                "15分足が上昇トレンド",
                "直近5分足の構造安値を損切りに使用",
            ],
        )
        plan["status"] = "買い待機" if plan.get("valid") else "見送り"
        plan["range_high"] = range_high
        plan["range_low"] = range_low
        return plan

    if row["TREND15"] == "下降":
        plan = calculate_plan(
            "SELL",
            range_low - TICK,
            recent_stop["High"].max() + STOP_BUFFER,
            f"{row['SESSION_KIND']}寄り後30分レンジ下抜け",
            [
                f"寄り後30分安値 {range_low:,}円を下抜け待ち",
                "15分足が下降トレンド",
                "直近5分足の構造高値を損切りに使用",
            ],
        )
        plan["status"] = "売り待機" if plan.get("valid") else "見送り"
        plan["range_high"] = range_high
        plan["range_low"] = range_low
        return plan

    return {
        "status": "見送り",
        "detail": "15分足がレンジのため、寄り後ブレイクを順張りで狙いません。",
        "range_high": range_high,
        "range_low": range_low,
    }


def choose_main_plan(pullback: dict, breakout: dict, current_price: float) -> dict:
    # 質の高い「15分足トレンド＋5分足25EMA押し目」を最優先
    for candidate in [pullback, breakout]:
        if candidate.get("valid"):
            plan = candidate.copy()
            if plan["direction"] == "BUY" and current_price >= plan["entry"] + max(TICK, plan["risk"] * 0.5):
                plan["status"] = "追いかけ注意"
                plan["extra_warning"] = "発動ラインからすでに離れています。次の押し目を待つ方が安全です。"
            elif plan["direction"] == "SELL" and current_price <= plan["entry"] - max(TICK, plan["risk"] * 0.5):
                plan["status"] = "追いかけ注意"
                plan["extra_warning"] = "発動ラインからすでに離れています。次の戻りを待つ方が安全です。"
            return plan
    return {
        "valid": False,
        "status": "見送り",
        "reason": "押し目・戻り売り、寄り後ブレイクのいずれも発動条件が揃っていません。",
    }


def calc_position_size(capital: float, plan: dict) -> dict:
    allowed_loss = int(capital * RISK_PER_TRADE_PCT)
    if not plan.get("valid"):
        return {"allowed_loss": allowed_loss, "contracts": 0, "risk_per_contract": None}

    risk_per_contract = int(plan["risk"] * YEN_PER_POINT_PER_MICRO)
    contracts = int(allowed_loss // risk_per_contract) if risk_per_contract > 0 else 0
    return {
        "allowed_loss": allowed_loss,
        "risk_per_contract": risk_per_contract,
        "contracts": contracts,
    }


def daily_reasons(row: pd.Series) -> list[str]:
    items = []
    items.append("終値が25日線より上" if row["Close"] > row["SMA25"] else "終値が25日線より下")
    items.append("25日線が75日線より上" if row["SMA25"] > row["SMA75"] else "25日線が75日線より下")
    items.append("日足MACDが上向き" if row["MACD"] > row["MACD_SIGNAL"] else "日足MACDが下向き")
    return items


# ------------------------
# 過去検証（簡易）
# ------------------------
def pullback_setup_at(df: pd.DataFrame, i: int) -> dict | None:
    if i < max(35, PULLBACK_STOP_BARS + 2):
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    recent = df.iloc[i - PULLBACK_TOUCH_BARS + 1 : i + 1]
    stop_window = df.iloc[i - PULLBACK_STOP_BARS + 1 : i + 1]

    if row["TREND15"] == "上昇":
        touch = (recent["Low"] <= recent["EMA25_5"] + TICK).any()
        reversal = (row["Close"] > row["Open"]) and (row["Close"] > prev["Close"]) and (row["Close"] > row["EMA25_5"])
        if touch and reversal:
            plan = calculate_plan("BUY", row["High"] + TICK, stop_window["Low"].min() - STOP_BUFFER, "pullback", [])
            if plan.get("valid"):
                return plan

    if row["TREND15"] == "下降":
        touch = (recent["High"] >= recent["EMA25_5"] - TICK).any()
        reversal = (row["Close"] < row["Open"]) and (row["Close"] < prev["Close"]) and (row["Close"] < row["EMA25_5"])
        if touch and reversal:
            plan = calculate_plan("SELL", row["Low"] - TICK, stop_window["High"].max() + STOP_BUFFER, "pullback", [])
            if plan.get("valid"):
                return plan
    return None


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
                # 同一足で損切りにも触れていたら、保守的に損切り扱い
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

    # 決着しない場合は、最終足の終値でR換算して評価する
    end_bar = df.iloc[min(last_i, len(df) - 1)]
    if direction == "BUY":
        result_r = (float(end_bar["Close"]) - entry) / plan["risk"]
    else:
        result_r = (entry - float(end_bar["Close"])) / plan["risk"]
    return float(np.clip(result_r, -1.0, TARGET_R1)), last_i


def backtest_pullback(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    i = 40
    while i < len(df) - 2:
        plan = pullback_setup_at(df, i)
        if plan is None:
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
    "15分足の方向に合わせ、5分足25EMAの押し目・戻り売りと寄り後レンジブレイクだけを狙うための判定アプリ"
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

# 現在値は最後の閉じた可能性が高い足で確認
current_i = get_last_completed_index(intraday)
current_row = intraday.iloc[current_i]
current_price = float(current_row["Close"])
latest_daily = daily.iloc[-1]

pullback = detect_pullback_setup(intraday)
breakout = detect_opening_breakout_setup(intraday)
main_plan = choose_main_plan(pullback, breakout, current_price)
position = calc_position_size(float(capital), main_plan)

last_bar_time = intraday.index[current_i]
minutes_old = max(0, int((pd.Timestamp.now(tz=JST) - last_bar_time).total_seconds() // 60))

# 9時間レンジを常に表示
range_window = intraday.iloc[max(0, current_i - RANGE_REFERENCE_BARS + 1) : current_i + 1]
range_high_9h = ceil_tick(range_window["High"].max())
range_low_9h = floor_tick(range_window["Low"].min())


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["総合判定", "セットアップ", "実行プラン", "簡易検証", "運用ルール"]
)

with tab1:
    st.subheader("今はどちら側だけを狙うか")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("日足の週方向", latest_daily["WEEK_SIGNAL"])
    c2.metric("15分足", current_row["TREND15"])
    c3.metric("5分足終値", f"{current_price:,.0f}")
    c4.metric("5分足の最終時刻", last_bar_time.strftime("%m/%d %H:%M"))

    if minutes_old > 20:
        st.warning(f"5分足が約{minutes_old}分前で止まっています。休場・データ遅延時は新規判断をしません。")

    if main_plan.get("valid"):
        if main_plan["direction"] == "BUY":
            st.success(f"総合判定：買い待機｜{main_plan['setup']}")
        else:
            st.error(f"総合判定：売り待機｜{main_plan['setup']}")
    else:
        st.warning("総合判定：見送り｜順張りの条件が揃うまで待機")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("日足の背景")
        for reason in daily_reasons(latest_daily):
            st.write(f"・{reason}")
    with right:
        st.subheader("現在の5分足")
        st.write(f"・セッション：{current_row['SESSION_KIND']}")
        st.write(f"・EMA9 / EMA25：{current_row['EMA9_5']:,.0f} / {current_row['EMA25_5']:,.0f}")
        if pd.notna(current_row["VWAP"]):
            relation = "上" if current_price > current_row["VWAP"] else "下"
            st.write(f"・VWAP：{current_row['VWAP']:,.0f}（終値はVWAPより{relation}）")
        else:
            st.write("・VWAP：参照データの出来高不足で未判定")
        st.write(f"・約9時間レンジ：{range_low_9h:,} 〜 {range_high_9h:,}")

    st.subheader("直近の5分足チャート")
    chart = intraday[["Close", "EMA9_5", "EMA25_5"]].tail(180).copy()
    if "VWAP" in intraday.columns and intraday["VWAP"].notna().any():
        chart["VWAP"] = intraday["VWAP"].tail(180)
    st.line_chart(chart)

with tab2:
    st.subheader("手法ごとのセットアップ判定")

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
        "優先順位は①押し目・戻り売り → ②寄り後ブレイクです。両方が揃わない場面は『何もしない』ことがルールです。"
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
            "発動ライン到達前に成行で入らない",
            "損切り注文を同時に置ける",
            "発動後に損切り幅が100円を超えていない",
            "直近で大きな経済イベント・要人発言の時間ではない",
        ]
        for check in checks:
            st.write(f"□ {check}")

with tab4:
    st.subheader("15分足トレンド＋5分足25EMA押し目／戻り売り｜簡易検証")
    st.caption(
        "直近30日程度の無料5分足で、利確1.5Rと損切り1Rのどちらが先かを保守的に集計します。"
    )

    if st.button("過去30日を検証する", use_container_width=True):
        try:
            with st.spinner("過去30日分の5分足を取得し、簡易検証しています..."):
                history = add_intraday_indicators(load_intraday("30d"))
                trades = backtest_pullback(history)
                st.session_state["backtest_trades"] = trades
        except Exception as error:
            st.error(f"検証データの取得に失敗しました：{error}")

    trades = st.session_state.get("backtest_trades")
    if isinstance(trades, pd.DataFrame):
        summary = summarize_backtest(trades)
        if summary is None:
            st.warning("条件に合うトレードが見つかりませんでした。期間が短いか、条件が厳しすぎる可能性があります。")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("トレード数", f"{summary['trades']}回")
            c2.metric("勝率", f"{summary['win_rate']:.1f}%")
            c3.metric("平均R", f"{summary['avg_r']:+.2f}R")
            c4.metric("最大連敗", f"{summary['max_losing_streak']}回")
            c5.metric("最大DD", f"{summary['max_drawdown_r']:.2f}R")

            shown = trades.copy()
            shown["日時"] = shown["日時"].dt.strftime("%m/%d %H:%M")
            shown["結果R"] = shown["結果R"].map(lambda x: f"{x:+.2f}R")
            st.dataframe(shown.tail(100), use_container_width=True, hide_index=True)

            st.warning(
                "この検証は無料データの簡易版です。手数料、スリッページ、日経225マイクロとの価格差、同一足内の約定順は厳密に反映していません。"
            )
    else:
        st.info("上の『過去30日を検証する』を押すと、簡易検証結果を表示します。")

with tab5:
    st.subheader("このアプリで守るルール")
    st.write("1. 15分足がレンジなら、5分足だけを見て逆張りしない。")
    st.write("2. 買いは15分足上昇＋5分足25EMA押し目、売りはその逆だけを狙う。")
    st.write("3. 寄り後ブレイクは最初の約2時間まで。遅い時間の追いかけはしない。")
    st.write("4. 損切りが100円を超える形は、期待値があっても見送る。")
    st.write("5. 1回の許容損失は資金の1%。連敗しても枚数を増やさない。")
    st.write("6. アプリは候補を出すだけ。発注はマネックスの実際の価格・チャートで最終確認する。")
