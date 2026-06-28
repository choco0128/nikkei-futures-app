import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="日経225先物 判定",
    page_icon="📈",
    layout="wide",
)

TICKER = "^N225"
TICKER_NAME = "日経225指数（週方向の基準）"


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data():
    data = yf.download(
        TICKER,
        period="10y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError("価格データを取得できませんでした。")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(col).strip().title() for col in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]

    if "Close" not in data.columns:
        raise RuntimeError("終値データが見つかりませんでした。")

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Close"])

    return data


def add_indicators(data):
    df = data.copy()
    close = df["Close"]

    df["SMA25"] = close.rolling(25).mean()
    df["SMA75"] = close.rolling(75).mean()

    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["20日高値"] = close.shift(1).rolling(20).max()
    df["20日安値"] = close.shift(1).rolling(20).min()

    df["トレンド点"] = np.where(close > df["SMA25"], 1, -1)
    df["移動平均点"] = np.where(df["SMA25"] > df["SMA75"], 1, -1)
    df["MACD点"] = np.where(df["MACD"] > df["MACD_SIGNAL"], 1, -1)

    df["RSI点"] = np.select(
        [
            (df["RSI"] >= 52) & (df["RSI"] <= 70),
            df["RSI"] < 48,
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    df["高安値点"] = np.select(
        [
            close > df["20日高値"],
            close < df["20日安値"],
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    df["合計点"] = (
        df["トレンド点"]
        + df["移動平均点"]
        + df["MACD点"]
        + df["RSI点"]
        + df["高安値点"]
    )

    df["判定"] = np.select(
        [
            df["合計点"] >= 3,
            df["合計点"] <= -3,
        ],
        [
            "買い優勢",
            "売り優勢",
        ],
        default="見送り",
    )

    df["5営業日後の変動率"] = df["Close"].shift(-5) / df["Close"] - 1

    return df.dropna(subset=["SMA75", "MACD_SIGNAL", "RSI"])


def get_stats(df, signal_name):
    history = df.iloc[:-5].copy()
    history = history[history["判定"] == signal_name].dropna(
        subset=["5営業日後の変動率"]
    )

    if history.empty:
        return None

    if signal_name == "買い優勢":
        trade_return = history["5営業日後の変動率"]
    else:
        trade_return = -history["5営業日後の変動率"]

    return {
        "回数": len(trade_return),
        "勝率": (trade_return > 0).mean() * 100,
        "平均変動率": trade_return.mean() * 100,
        "中央値": trade_return.median() * 100,
        "最大不利変動": trade_return.min() * 100,
    }


def build_reasons(row):
    positive = []
    caution = []

    if row["Close"] > row["SMA25"]:
        positive.append("終値が25日線より上")
    else:
        caution.append("終値が25日線より下")

    if row["SMA25"] > row["SMA75"]:
        positive.append("25日線が75日線より上")
    else:
        caution.append("25日線が75日線より下")

    if row["MACD"] > row["MACD_SIGNAL"]:
        positive.append("MACDがシグナルより上")
    else:
        caution.append("MACDがシグナルより下")

    if 52 <= row["RSI"] <= 70:
        positive.append("RSIが上昇に適した範囲")
    elif row["RSI"] > 70:
        caution.append("RSIが70超で短期的に過熱気味")
    else:
        caution.append("RSIが弱い水準")

    if row["Close"] > row["20日高値"]:
        positive.append("20日高値を上抜け")
    elif row["Close"] < row["20日安値"]:
        caution.append("20日安値を下抜け")

    return positive, caution


def format_percent(value):
    return f"{value:+.2f}%"


st.title("日経225先物｜週方向判定")
st.caption("日経225指数の日足を使い、次の5営業日の方向を過去データから統計的に判定します。")

st.warning(
    "このVer.1は日経225指数の日足ベースです。"
    "夜間先物・寄り付きギャップ・5分足はまだ反映していません。"
)

if st.button("データを更新"):
    st.cache_data.clear()

try:
    with st.spinner("過去データを読み込んでいます..."):
        raw_data = load_price_data()
        df = add_indicators(raw_data)

except Exception as error:
    st.error(f"データ取得でエラーが出ました：{error}")
    st.stop()

latest = df.iloc[-1]
last_date = df.index[-1].strftime("%Y年%m月%d日")
signal = latest["判定"]

buy_stats = get_stats(df, "買い優勢")
sell_stats = get_stats(df, "売り優勢")

signal_strength = min(100, int(abs(latest["合計点"]) / 5 * 100))
positive_reasons, caution_reasons = build_reasons(latest)

tab1, tab2, tab3 = st.tabs(["週方向判定", "過去検証", "使い方"])

with tab1:
    st.subheader("現在の週方向判定")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("判定", signal)

    with col2:
        st.metric("ルール一致度", f"{signal_strength}点")

    with col3:
        st.metric("最終データ日", last_date)

    if signal == "買い優勢":
        st.success("買い優勢です。ただし実際の買いは、翌週の値動きと損切り幅を確認してから判断します。")
        current_stats = buy_stats
    elif signal == "売り優勢":
        st.error("売り優勢です。戻り売りを検討できる地合いですが、逆行時の損切り位置を先に決めます。")
        current_stats = sell_stats
    else:
        st.warning("買い・売りの条件が揃っていません。無理に入らないための見送り判定です。")
        current_stats = None

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("買い優勢の根拠")
        if positive_reasons:
            for reason in positive_reasons:
                st.write(f"・{reason}")
        else:
            st.write("・強い買い材料は揃っていません。")

    with right:
        st.subheader("注意点・売り材料")
        if caution_reasons:
            for reason in caution_reasons:
                st.write(f"・{reason}")
        else:
            st.write("・大きな注意点は検出されていません。")

    st.divider()

    st.subheader("指標の現在値")

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("終値", f"{latest['Close']:,.0f}")
    metric2.metric("25日線", f"{latest['SMA25']:,.0f}")
    metric3.metric("75日線", f"{latest['SMA75']:,.0f}")
    metric4.metric("RSI", f"{latest['RSI']:.1f}")

    st.divider()

    if current_stats:
        st.subheader("現在と同じ判定が出た過去の結果")
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("該当回数", f"{current_stats['回数']}回")
        stat2.metric("5営業日後の勝率", f"{current_stats['勝率']:.1f}%")
        stat3.metric("平均変動率", format_percent(current_stats["平均変動率"]))

        st.caption(
            "勝率や平均変動率は将来の利益を保証しません。"
            "実際の取引ではエントリー位置、損切り幅、ロット管理を別途確認します。"
        )

    st.subheader("直近の値動き")
    chart_data = df[["Close", "SMA25", "SMA75"]].tail(180)
    st.line_chart(chart_data)

with tab2:
    st.subheader("過去10年の簡易検証")
    st.caption("判定が出た日の終値から、5営業日後まで保有した場合の変動を集計しています。")

    rows = []

    for label, stats in [("買い優勢", buy_stats), ("売り優勢", sell_stats)]:
        if stats:
            rows.append(
                {
                    "判定": label,
                    "該当回数": stats["回数"],
                    "勝率": f"{stats['勝率']:.1f}%",
                    "平均変動率": format_percent(stats["平均変動率"]),
                    "中央値": format_percent(stats["中央値"]),
                    "最大不利変動": format_percent(stats["最大不利変動"]),
                }
            )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("検証結果を作成中です。")

    st.warning(
        "この検証は手数料・スリッページ・夜間ギャップ・損切りを入れていない簡易版です。"
        "次の段階で、実際の損切り幅と利確幅を使ったR倍数ベースの検証に進めます。"
    )

with tab3:
    st.subheader("このアプリの見方")

    st.write("1. 買い優勢でも、すぐに買うとは限りません。")
    st.write("2. 実際には、翌週の高値更新や押し目などのエントリー条件を使います。")
    st.write("3. 見送りは失敗ではなく、優位性が弱い局面を避けるための判定です。")
    st.write("4. 次の更新で、エントリーライン・損切りライン・許容ロットを追加します。")