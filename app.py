import streamlit as st

# AUTO_5MIN_V4 — 直近高値・安値の手入力なし
import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# 日経225先物｜5分足自動取得版
# 参照データ: Yahoo Finance の NIY=F（CME Nikkei/Yen Futures）
# 注意: 日経225マイクロそのものではないため、発注は必ず証券会社画面で最終確認する。
# ============================================================

st.set_page_config(page_title="日経225先物｜自動5分足判定", page_icon="📈", layout="wide")

TICKER_DAILY = "^N225"
TICKER_INTRADAY = "NIY=F"
JST = "Asia/Tokyo"
TICK = 5
YEN_PER_POINT_PER_MICRO = 10


def clean_columns(data: pd.DataFrame) -> pd.DataFrame:
    """yfinanceの列を通常のOHLCV形式にそろえる。"""
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_jst_index(df: pd.DataFrame) -> pd.DataFrame:
    """インデックスを日本時間にそろえる。"""
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC").tz_convert(JST)
    else:
        idx = idx.tz_convert(JST)
    out.index = idx
    return out


@st.cache_data(ttl=60, show_spinner=False)
def load_intraday() -> pd.DataFrame:
    """直近5日間の5分足を自動取得する。"""
    data = yf.download(
        TICKER_INTRADAY,
        period="5d",
        interval="5m",
        auto_adjust=False,
        prepost=True,
        progress=False,
        threads=False,
    )
    df = clean_columns(data)
    required = {"Open", "High", "Low", "Close"}
    if df.empty or not required.issubset(df.columns):
        raise RuntimeError("5分足を取得できませんでした。市場休場・Yahoo側の一時的な不具合・ティッカーの更新が考えられます。")

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df.empty:
        raise RuntimeError("利用できる5分足がありません。")

    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["Volume"] = df["Volume"].fillna(0).clip(lower=0)
    return to_jst_index(df).sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily() -> pd.DataFrame:
    """週方向の確認用に日経225の日足を自動取得する。"""
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


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["SMA25"] = c.rolling(25).mean()
    out["SMA75"] = c.rolling(75).mean()
    out["EMA12"] = c.ewm(span=12, adjust=False).mean()
    out["EMA26"] = c.ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    out["POINT"] = (
        np.where(c > out["SMA25"], 1, -1)
        + np.where(out["SMA25"] > out["SMA75"], 1, -1)
        + np.where(out["MACD"] > out["MACD_SIGNAL"], 1, -1)
        + np.select([(out["RSI"] >= 52) & (out["RSI"] <= 70), out["RSI"] < 48], [1, -1], default=0)
    )
    out["WEEKLY_SIGNAL"] = np.select(
        [out["POINT"] >= 3, out["POINT"] <= -3],
        ["買い優勢", "売り優勢"],
        default="見送り",
    )
    return out.dropna(subset=["SMA75", "MACD_SIGNAL", "RSI"])


def session_key(idx: pd.DatetimeIndex) -> pd.Index:
    """17:00以降の夜間を翌取引日側に寄せる。"""
    return pd.Index((idx + pd.Timedelta(hours=7)).date)


def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["EMA9"] = c.ewm(span=9, adjust=False).mean()
    out["EMA20"] = c.ewm(span=20, adjust=False).mean()
    out["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()

    prev_close = c.shift(1)
    tr = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()

    out["SESSION"] = session_key(out.index)
    if out["Volume"].sum() > 0:
        typical = (out["High"] + out["Low"] + out["Close"]) / 3
        cumulative_pv = (typical * out["Volume"]).groupby(out["SESSION"]).cumsum()
        cumulative_volume = out["Volume"].groupby(out["SESSION"]).cumsum()
        out["VWAP"] = cumulative_pv / cumulative_volume.replace(0, np.nan)
    else:
        out["VWAP"] = np.nan

    return out.dropna(subset=["EMA20", "MACD_SIGNAL", "ATR14"]).copy()


def ceil_tick(value: float) -> int:
    return int(np.ceil(float(value) / TICK) * TICK)


def floor_tick(value: float) -> int:
    return int(np.floor(float(value) / TICK) * TICK)


def minutes_old(ts: pd.Timestamp) -> int:
    now = pd.Timestamp.now(tz=JST)
    return max(0, int((now - ts).total_seconds() // 60))


def weekly_reasons(row: pd.Series) -> tuple[list[str], list[str]]:
    bull, bear = [], []
    if row["Close"] > row["SMA25"]:
        bull.append("終値が25日線より上")
    else:
        bear.append("終値が25日線より下")
    if row["SMA25"] > row["SMA75"]:
        bull.append("25日線が75日線より上")
    else:
        bear.append("25日線が75日線より下")
    if row["MACD"] > row["MACD_SIGNAL"]:
        bull.append("日足MACDがシグナルより上")
    else:
        bear.append("日足MACDがシグナルより下")
    if 52 <= row["RSI"] <= 70:
        bull.append("RSIが上昇に適した範囲")
    elif row["RSI"] > 70:
        bear.append("RSIが70超で短期過熱")
    else:
        bear.append("RSIが弱い水準")
    return bull, bear


def auto_plan(intra: pd.DataFrame, lookback: int, price_adjustment: int, stop_buffer: int, rr: float) -> dict:
    """手入力なしで直近5分足からブレイク計画を作る。"""
    if len(intra) < lookback + 2:
        raise ValueError("5分足が不足しています。少し待ってから更新してください。")

    latest = intra.iloc[-1]
    prior = intra.iloc[-(lookback + 1):-1]
    raw_high = float(prior["High"].max())
    raw_low = float(prior["Low"].min())

    range_high = ceil_tick(raw_high + price_adjustment)
    range_low = floor_tick(raw_low + price_adjustment)

    buy_entry = ceil_tick(range_high + TICK)
    sell_entry = floor_tick(range_low - TICK)
    buy_stop = floor_tick(range_low - stop_buffer)
    sell_stop = ceil_tick(range_high + stop_buffer)

    buy_risk = buy_entry - buy_stop
    sell_risk = sell_stop - sell_entry
    buy_target = ceil_tick(buy_entry + buy_risk * rr)
    sell_target = floor_tick(sell_entry - sell_risk * rr)

    bull, bear, score = [], [], 0
    if pd.notna(latest["VWAP"]):
        if latest["Close"] > latest["VWAP"]:
            bull.append("終値がVWAPより上")
            score += 1
        else:
            bear.append("終値がVWAPより下")
            score -= 1
    else:
        bear.append("出来高不足でVWAPは未判定")

    if latest["EMA9"] > latest["EMA20"]:
        bull.append("EMA9がEMA20より上")
        score += 1
    else:
        bear.append("EMA9がEMA20より下")
        score -= 1

    if latest["MACD"] > latest["MACD_SIGNAL"]:
        bull.append("5分足MACDがシグナルより上")
        score += 1
    else:
        bear.append("5分足MACDがシグナルより下")
        score -= 1

    if latest["Close"] >= buy_entry:
        bull.append("直近レンジ高値を上抜け済み")
        score += 1
    elif latest["Close"] <= sell_entry:
        bear.append("直近レンジ安値を下抜け済み")
        score -= 1

    if score >= 2:
        bias = "買い条件寄り"
    elif score <= -2:
        bias = "売り条件寄り"
    else:
        bias = "中立・待機"

    return {
        "latest": latest,
        "range_high": range_high,
        "range_low": range_low,
        "buy_entry": buy_entry,
        "buy_stop": buy_stop,
        "buy_target": buy_target,
        "buy_risk": buy_risk,
        "sell_entry": sell_entry,
        "sell_stop": sell_stop,
        "sell_target": sell_target,
        "sell_risk": sell_risk,
        "bull": bull,
        "bear": bear,
        "score": score,
        "bias": bias,
    }


def position_size(stop_points: int, account: int, risk_pct: float, daily_limit_pct: float, today_pnl: int) -> dict:
    risk_budget = account * risk_pct / 100
    daily_limit = account * daily_limit_pct / 100
    remaining = max(0, daily_limit - max(0, -today_pnl))
    usable = min(risk_budget, remaining)
    loss_per_contract = stop_points * YEN_PER_POINT_PER_MICRO
    contracts = int(usable // loss_per_contract) if loss_per_contract > 0 else 0
    return {
        "risk_budget": risk_budget,
        "daily_limit": daily_limit,
        "remaining": remaining,
        "usable": usable,
        "loss_per_contract": loss_per_contract,
        "contracts": contracts,
        "max_loss": contracts * loss_per_contract,
    }


# ============================================================
# UI
# ============================================================
st.title("日経225先物｜自動5分足判定 v4")
st.caption("直近高値・安値の手入力なし。5分足を自動取得し、ブレイク候補を自動計算します。")
st.info(
    "参照価格はYahoo FinanceのNIY=F（CMEの日経円建て先物）です。"
    "日経225マイクロとは一致しないことがあるため、発注前にマネックスの5分足・価格で最終確認してください。"
)

update_col, settings_col = st.columns([1, 2])
with update_col:
    if st.button("自動データを更新（日足＋5分足）", type="primary"):
        st.cache_data.clear()
with settings_col:
    lookback = st.select_slider(
        "自動レンジに使う本数（5分足）",
        options=[6, 9, 12, 18, 24, 36],
        value=12,
        help="12本＝直近60分の高値・安値をブレイクラインに使います。",
    )

with st.sidebar:
    st.header("設定")
    st.caption("高値・安値の入力は不要です。必要なら価格差だけ補正します。")
    price_adjustment = st.number_input(
        "マネックスとの価格差補正（円）",
        value=0,
        step=TICK,
        help="自動データよりマネックス先物が10円高いときは +10。通常は0のままです。",
    )
    stop_buffer = st.number_input("損切りバッファ（円）", min_value=TICK, value=10, step=TICK)
    reward_risk = st.selectbox("目標リスクリワード", [1.0, 1.2, 1.5, 2.0], index=2)
    st.divider()
    account_balance = st.number_input("口座資金（円）", min_value=10_000, value=190_000, step=10_000)
    risk_percent = st.number_input("1回の許容損失（資金比 %）", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    daily_limit_percent = st.number_input("1日の損失上限（資金比 %）", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    today_pnl = st.number_input("今日の確定損益（円）", value=0, step=500, help="損失はマイナスで入力します。")

try:
    with st.spinner("日足と5分足を自動取得しています..."):
        daily = add_daily_indicators(load_daily())
        intraday = add_intraday_indicators(load_intraday())
        plan = auto_plan(intraday, int(lookback), int(price_adjustment), int(stop_buffer), float(reward_risk))
except Exception as e:
    st.error(f"データ取得エラー：{e}")
    st.stop()

weekly = daily.iloc[-1]
latest = plan["latest"]
weekly_bull, weekly_bear = weekly_reasons(weekly)

if weekly["WEEKLY_SIGNAL"] == "買い優勢" and plan["bias"] == "買い条件寄り":
    overall = "買い候補"
    overall_text = "日足と5分足が買い側にそろっています。5分足終値で上抜け確認後だけ検討します。"
elif weekly["WEEKLY_SIGNAL"] == "売り優勢" and plan["bias"] == "売り条件寄り":
    overall = "売り候補"
    overall_text = "日足と5分足が売り側にそろっています。5分足終値で下抜け確認後だけ検討します。"
elif plan["bias"] == "中立・待機":
    overall = "見送り"
    overall_text = "5分足の方向が定まっていません。上下どちらかのブレイクを待つ局面です。"
else:
    overall = "方向不一致・見送り"
    overall_text = "日足と5分足の向きがそろっていません。無理なエントリーを避けます。"

buy_size = position_size(plan["buy_risk"], int(account_balance), float(risk_percent), float(daily_limit_percent), int(today_pnl))
sell_size = position_size(plan["sell_risk"], int(account_balance), float(risk_percent), float(daily_limit_percent), int(today_pnl))

if minutes_old(latest.name) > 30:
    st.warning(f"自動5分足の最終時刻から約{minutes_old(latest.name)}分経過しています。市場休場またはデータ遅延の可能性があります。発注には使わず、次の取引前に更新してください。")

tab1, tab2, tab3, tab4 = st.tabs(["総合判定", "自動5分足", "自動実行プラン", "資金管理"])

with tab1:
    st.subheader("次セッションに向けた判定")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総合判定", overall)
    c2.metric("週方向", weekly["WEEKLY_SIGNAL"])
    c3.metric("5分足", plan["bias"])
    c4.metric("5分足最終時刻", latest.name.strftime("%m/%d %H:%M"))

    if "買い候補" in overall:
        st.success(overall_text)
    elif "売り候補" in overall:
        st.error(overall_text)
    else:
        st.warning(overall_text)

    st.subheader("自動ブレイクライン")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("買い上抜け", f"{plan['buy_entry']:,}")
    b2.metric("売り下抜け", f"{plan['sell_entry']:,}")
    b3.metric("自動レンジ高値", f"{plan['range_high']:,}")
    b4.metric("自動レンジ安値", f"{plan['range_low']:,}")
    st.caption(f"直近{lookback}本（約{int(lookback) * 5}分）の最新バーを除いた高値・安値から自動計算。")

    left, right = st.columns(2)
    with left:
        st.subheader("日足の買い材料")
        for text in weekly_bull or ["強い買い材料はありません。"]:
            st.write(f"・{text}")
    with right:
        st.subheader("日足の注意点")
        for text in weekly_bear or ["大きな注意点はありません。"]:
            st.write(f"・{text}")

with tab2:
    st.subheader("自動取得した5分足")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("直近終値", f"{latest['Close']:,.0f}")
    c2.metric("VWAP", "--" if pd.isna(latest["VWAP"]) else f"{latest['VWAP']:,.0f}")
    c3.metric("ATR(14)", f"{latest['ATR14']:.1f}")
    c4.metric("短期スコア", f"{plan['score']:+d}")

    left, right = st.columns(2)
    with left:
        st.subheader("買い側の根拠")
        for text in plan["bull"] or ["買い側の条件はまだ少ないです。"]:
            st.write(f"・{text}")
    with right:
        st.subheader("売り側の根拠・注意")
        for text in plan["bear"] or ["大きな売り材料はありません。"]:
            st.write(f"・{text}")

    chart_cols = ["Close", "EMA9", "EMA20"]
    if intraday["VWAP"].notna().any():
        chart_cols.append("VWAP")
    st.line_chart(intraday[chart_cols].tail(180))

    with st.expander("直近30本の5分足を確認"):
        show = intraday[["Open", "High", "Low", "Close", "Volume", "EMA9", "EMA20", "MACD", "MACD_SIGNAL", "VWAP"]].tail(30).copy()
        show.index = show.index.strftime("%Y-%m-%d %H:%M")
        st.dataframe(show.round(2), use_container_width=True)

with tab3:
    st.subheader("自動実行プラン")
    st.caption("直近高値・安値は入力不要です。自動5分足レンジから作っています。")

    buy_col, sell_col = st.columns(2)
    with buy_col:
        st.markdown("### 買いプラン")
        st.metric("エントリー", f"{plan['buy_entry']:,}")
        st.metric("損切り", f"{plan['buy_stop']:,}")
        st.metric("第一利確", f"{plan['buy_target']:,}")
        st.metric("損切り幅", f"{plan['buy_risk']:,}円")
        st.metric("想定RR", f"{reward_risk:.1f}")
        if plan["buy_risk"] > 100:
            st.warning("損切り幅が100円を超えています。あなたの通常ルールでは見送り候補です。")
        else:
            st.info("5分足終値で上抜け確認後のみ検討。上抜け前の先回りはしません。")

    with sell_col:
        st.markdown("### 売りプラン")
        st.metric("エントリー", f"{plan['sell_entry']:,}")
        st.metric("損切り", f"{plan['sell_stop']:,}")
        st.metric("第一利確", f"{plan['sell_target']:,}")
        st.metric("損切り幅", f"{plan['sell_risk']:,}円")
        st.metric("想定RR", f"{reward_risk:.1f}")
        if plan["sell_risk"] > 100:
            st.warning("損切り幅が100円を超えています。あなたの通常ルールでは見送り候補です。")
        else:
            st.info("5分足終値で下抜け確認後のみ検討。下抜け前の先回りはしません。")

with tab4:
    st.subheader("資金管理")
    st.caption("日経225マイクロは1ポイントあたり1枚10円で計算しています。")

    daily_loss_limit = int(account_balance * daily_limit_percent / 100)
    remaining_daily = max(0, daily_loss_limit - max(0, -int(today_pnl)))
    s1, s2, s3 = st.columns(3)
    s1.metric("1回の許容損失", f"{int(account_balance * risk_percent / 100):,}円")
    s2.metric("1日の損失上限", f"{daily_loss_limit:,}円")
    s3.metric("今日の残り損失枠", f"{remaining_daily:,}円")

    buy_col, sell_col = st.columns(2)
    with buy_col:
        st.markdown("### 買い時の許容枚数")
        st.metric("1枚あたり損失", f"{int(buy_size['loss_per_contract']):,}円")
        st.metric("最大枚数", f"{buy_size['contracts']}枚")
        st.metric("最大想定損失", f"{int(buy_size['max_loss']):,}円")
    with sell_col:
        st.markdown("### 売り時の許容枚数")
        st.metric("1枚あたり損失", f"{int(sell_size['loss_per_contract']):,}円")
        st.metric("最大枚数", f"{sell_size['contracts']}枚")
        st.metric("最大想定損失", f"{int(sell_size['max_loss']):,}円")

    if remaining_daily <= 0:
        st.error("本日の損失上限に到達しています。新規取引は停止です。")
    elif buy_size["contracts"] == 0 and sell_size["contracts"] == 0:
        st.warning("現在の損切り幅では、許容損失内で1枚も入れません。見送りです。")
    else:
        st.success("損失上限内の枚数です。ただし、実際の発注はマネックス価格・板・5分足終値で最終確認してから行います。")

st.divider()
st.caption("このアプリは売買推奨ではありません。無料参照データのため、実際のマイクロ先物価格・板・約定状況とは差が出ることがあります。")
