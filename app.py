
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="日経225先物 判定",
    page_icon="📈",
    layout="wide",
)

# 日経225マイクロ先物：1円の変動 = 10円 / 枚、呼値は5円刻み
TICK_SIZE = 5
YEN_PER_POINT_PER_CONTRACT = 10
TICKER = "^N225"


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
    data = data.loc[:, ~data.columns.duplicated()].copy()

    required = {"Close", "High", "Low"}
    if not required.issubset(data.columns):
        raise RuntimeError("終値・高値・安値データが見つかりませんでした。")

    for col in ["Close", "High", "Low"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(subset=["Close", "High", "Low"])


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

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR14"] = true_range.rolling(14).mean()

    df["20日高値"] = close.shift(1).rolling(20).max()
    df["20日安値"] = close.shift(1).rolling(20).min()

    df["トレンド点"] = np.where(close > df["SMA25"], 1, -1)
    df["移動平均点"] = np.where(df["SMA25"] > df["SMA75"], 1, -1)
    df["MACD点"] = np.where(df["MACD"] > df["MACD_SIGNAL"], 1, -1)
    df["RSI点"] = np.select(
        [(df["RSI"] >= 52) & (df["RSI"] <= 70), df["RSI"] < 48],
        [1, -1],
        default=0,
    )
    df["高安値点"] = np.select(
        [close > df["20日高値"], close < df["20日安値"]],
        [1, -1],
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
        [df["合計点"] >= 3, df["合計点"] <= -3],
        ["買い優勢", "売り優勢"],
        default="見送り",
    )

    df["5営業日後の変動率"] = df["Close"].shift(-5) / df["Close"] - 1

    return df.dropna(subset=["SMA75", "MACD_SIGNAL", "RSI", "ATR14"])


def get_stats(df, signal_name):
    history = df.iloc[:-5].copy()
    history = history[history["判定"] == signal_name].dropna(
        subset=["5営業日後の変動率"]
    )

    if history.empty:
        return None

    returns = history["5営業日後の変動率"]
    trade_return = returns if signal_name == "買い優勢" else -returns
    win_loss = (trade_return > 0).tolist()

    longest_loss_streak = 0
    current_loss_streak = 0
    for is_win in win_loss:
        if is_win:
            current_loss_streak = 0
        else:
            current_loss_streak += 1
            longest_loss_streak = max(longest_loss_streak, current_loss_streak)

    return {
        "回数": len(trade_return),
        "勝率": (trade_return > 0).mean() * 100,
        "平均変動率": trade_return.mean() * 100,
        "中央値": trade_return.median() * 100,
        "最大不利変動": trade_return.min() * 100,
        "最大連敗": longest_loss_streak,
    }


def build_reasons(row):
    positives = []
    cautions = []

    if row["Close"] > row["SMA25"]:
        positives.append("終値が25日線より上")
    else:
        cautions.append("終値が25日線より下")

    if row["SMA25"] > row["SMA75"]:
        positives.append("25日線が75日線より上")
    else:
        cautions.append("25日線が75日線より下")

    if row["MACD"] > row["MACD_SIGNAL"]:
        positives.append("MACDがシグナルより上")
    else:
        cautions.append("MACDがシグナルより下")

    if 52 <= row["RSI"] <= 70:
        positives.append("RSIが上昇に適した範囲")
    elif row["RSI"] > 70:
        cautions.append("RSIが70超で短期的に過熱気味")
    else:
        cautions.append("RSIが弱い水準")

    if row["Close"] > row["20日高値"]:
        positives.append("20日高値を上抜け")
    elif row["Close"] < row["20日安値"]:
        cautions.append("20日安値を下抜け")

    return positives, cautions


def round_up_to_tick(value):
    return int(np.ceil(value / TICK_SIZE) * TICK_SIZE)


def round_down_to_tick(value):
    return int(np.floor(value / TICK_SIZE) * TICK_SIZE)


def make_trade_plan(direction, range_high, range_low, stop_buffer, reward_risk):
    if range_high <= range_low:
        raise ValueError("直近高値は直近安値より大きくしてください。")

    if direction == "買い":
        entry = round_up_to_tick(range_high + TICK_SIZE)
        stop = round_down_to_tick(range_low - stop_buffer)
        risk_points = entry - stop
        target = round_up_to_tick(entry + risk_points * reward_risk)
    else:
        entry = round_down_to_tick(range_low - TICK_SIZE)
        stop = round_up_to_tick(range_high + stop_buffer)
        risk_points = stop - entry
        target = round_down_to_tick(entry - risk_points * reward_risk)

    reward_points = abs(target - entry)

    return {
        "方向": direction,
        "エントリー": entry,
        "損切り": stop,
        "第一利確": target,
        "損切り幅": risk_points,
        "利確幅": reward_points,
        "RR": reward_points / risk_points if risk_points else 0,
    }


def plan_risk(plan, account_balance, risk_percent, daily_limit_percent, today_pnl):
    per_trade_budget = account_balance * risk_percent / 100
    daily_loss_limit = account_balance * daily_limit_percent / 100
    used_loss = max(0, -today_pnl)
    remaining_daily_budget = max(0, daily_loss_limit - used_loss)
    usable_budget = min(per_trade_budget, remaining_daily_budget)

    loss_per_contract = plan["損切り幅"] * YEN_PER_POINT_PER_CONTRACT
    profit_per_contract = plan["利確幅"] * YEN_PER_POINT_PER_CONTRACT
    contracts = int(usable_budget // loss_per_contract) if loss_per_contract > 0 else 0

    return {
        "1回の許容損失": per_trade_budget,
        "1日の損失上限": daily_loss_limit,
        "本日の残り損失枠": remaining_daily_budget,
        "実際に使う損失枠": usable_budget,
        "1枚あたり損失": loss_per_contract,
        "1枚あたり利確額": profit_per_contract,
        "最大枚数": contracts,
        "最大想定損失": contracts * loss_per_contract,
        "最大想定利益": contracts * profit_per_contract,
    }


def money(value):
    return f"{value:,.0f}円"


def percent(value):
    return f"{value:+.2f}%"


st.title("日経225先物｜実戦判定アプリ")
st.caption("週方向の偏り、5分足ブレイクの取引計画、損失上限・ロット管理を1画面で確認します。")

st.warning(
    "週方向は日経225指数の日足ベースです。実際の発注は、日経225マイクロ先物の"
    "5分足で見た直近高値・安値を入力してから行ってください。"
)

if st.button("日足データを更新"):
    st.cache_data.clear()

try:
    with st.spinner("日足データを読み込んでいます..."):
        raw = load_price_data()
        df = add_indicators(raw)
except Exception as error:
    st.error(f"データ取得でエラーが出ました：{error}")
    st.stop()

latest = df.iloc[-1]
signal = latest["判定"]
last_date = df.index[-1].strftime("%Y年%m月%d日")
signal_strength = min(100, int(abs(latest["合計点"]) / 5 * 100))
positive_reasons, caution_reasons = build_reasons(latest)
buy_stats = get_stats(df, "買い優勢")
sell_stats = get_stats(df, "売り優勢")

reference_price = round_up_to_tick(latest["Close"])
default_high = reference_price + 50
default_low = reference_price - 50

tab1, tab2, tab3, tab4 = st.tabs(
    ["週方向", "実行プラン", "過去検証", "運用ルール"]
)

with tab1:
    st.subheader("次の5営業日の方向判定")

    c1, c2, c3 = st.columns(3)
    c1.metric("判定", signal)
    c2.metric("ルール一致度", f"{signal_strength}点")
    c3.metric("最終データ日", last_date)

    if signal == "買い優勢":
        st.success("週方向は買い優勢です。次のセッションでは、5分足の上抜けだけを狙う前提です。")
        current_stats = buy_stats
    elif signal == "売り優勢":
        st.error("週方向は売り優勢です。次のセッションでは、5分足の下抜けだけを狙う前提です。")
        current_stats = sell_stats
    else:
        st.warning("週方向は見送りです。5分足の明確なブレイクと資金管理条件が揃うまで新規エントリーを急ぎません。")
        current_stats = None

    left, right = st.columns(2)
    with left:
        st.subheader("買い材料")
        if positive_reasons:
            for reason in positive_reasons:
                st.write(f"・{reason}")
        else:
            st.write("・強い買い材料は揃っていません。")

    with right:
        st.subheader("売り材料・注意点")
        if caution_reasons:
            for reason in caution_reasons:
                st.write(f"・{reason}")
        else:
            st.write("・大きな注意点は検出されていません。")

    st.subheader("日足の現在値")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("終値", f"{latest['Close']:,.0f}")
    i2.metric("25日線", f"{latest['SMA25']:,.0f}")
    i3.metric("75日線", f"{latest['SMA75']:,.0f}")
    i4.metric("RSI", f"{latest['RSI']:.1f}")

    st.caption(f"14日ATR：{latest['ATR14']:,.0f}円。ATRが大きい日は、普段より損切り幅が広がりやすい点に注意します。")

    if current_stats:
        st.subheader("同じ週方向が出た過去の結果")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("該当回数", f"{current_stats['回数']}回")
        s2.metric("5営業日後の勝率", f"{current_stats['勝率']:.1f}%")
        s3.metric("平均変動率", percent(current_stats["平均変動率"]))
        s4.metric("最大連敗", f"{current_stats['最大連敗']}回")

    st.subheader("直近180日")
    st.line_chart(df[["Close", "SMA25", "SMA75"]].tail(180))

with tab2:
    st.subheader("次セッションの取引計画")
    st.caption("日経225マイクロ先物の5分足で確認した直近高値・安値を入力してください。数値は5円刻みにそろえます。")

    with st.expander("入力：5分足のブレイク幅・資金管理", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            range_high = st.number_input(
                "直近高値（先物）",
                min_value=1000,
                value=int(default_high),
                step=TICK_SIZE,
            )
            range_low = st.number_input(
                "直近安値（先物）",
                min_value=1000,
                value=int(default_low),
                step=TICK_SIZE,
            )
            stop_buffer = st.number_input(
                "損切りバッファ（円）",
                min_value=TICK_SIZE,
                value=10,
                step=TICK_SIZE,
                help="直近安値・高値を少し抜けた位置に損切りを置くための余白です。",
            )
            max_stop_points = st.number_input(
                "許容する最大損切り幅（円）",
                min_value=20,
                value=100,
                step=TICK_SIZE,
            )

        with col_b:
            account_balance = st.number_input(
                "口座資金（円）",
                min_value=10000,
                value=200000,
                step=10000,
            )
            risk_percent = st.number_input(
                "1回の許容損失（資金比 %）",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
            )
            daily_limit_percent = st.number_input(
                "1日の損失上限（資金比 %）",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
            )
            today_pnl = st.number_input(
                "今日の確定損益（円）",
                value=0,
                step=500,
                help="損失の場合はマイナスで入力します。",
            )
            reward_risk = st.selectbox(
                "目標リスクリワード",
                options=[1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
                index=2,
            )

    if range_high <= range_low:
        st.error("直近高値は、直近安値より大きくしてください。")
    else:
        buy_plan = make_trade_plan("買い", range_high, range_low, stop_buffer, reward_risk)
        sell_plan = make_trade_plan("売り", range_high, range_low, stop_buffer, reward_risk)

        if signal == "買い優勢":
            primary_plan, secondary_plan = buy_plan, sell_plan
            primary_label = "順張りの買い計画"
            secondary_label = "逆張りの売り計画（原則見送り）"
        elif signal == "売り優勢":
            primary_plan, secondary_plan = sell_plan, buy_plan
            primary_label = "順張りの売り計画"
            secondary_label = "逆張りの買い計画（原則見送り）"
        else:
            primary_plan, secondary_plan = buy_plan, sell_plan
            primary_label = "買いブレイク計画"
            secondary_label = "売りブレイク計画"

        primary_risk = plan_risk(
            primary_plan,
            account_balance,
            risk_percent,
            daily_limit_percent,
            today_pnl,
        )

        st.subheader(primary_label)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("エントリー", f"{primary_plan['エントリー']:,}円")
        p2.metric("損切り", f"{primary_plan['損切り']:,}円")
        p3.metric("第一利確", f"{primary_plan['第一利確']:,}円")
        p4.metric("想定RR", f"{primary_plan['RR']:.1f}")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("損切り幅", f"{primary_plan['損切り幅']:,}円")
        q2.metric("1枚あたり損失", money(primary_risk["1枚あたり損失"]))
        q3.metric("許容枚数", f"{primary_risk['最大枚数']}枚")
        q4.metric("最大想定損失", money(primary_risk["最大想定損失"]))

        status_messages = []

        if today_pnl <= -primary_risk["1日の損失上限"]:
            status_messages.append("本日の損失上限に到達しています。新規取引を停止します。")
        if primary_plan["損切り幅"] > max_stop_points:
            status_messages.append(
                f"損切り幅が上限の{max_stop_points:,}円を超えています。見送りです。"
            )
        if primary_risk["最大枚数"] < 1:
            status_messages.append("許容損失の範囲では1枚も建てられません。見送りです。")
        if signal == "見送り":
            status_messages.append(
                "週方向が見送りです。5分足終値での明確なブレイク確認がない限り、見送ります。"
            )

        if status_messages:
            for message in status_messages:
                st.warning(message)
        else:
            st.success(
                f"取引条件は資金管理の範囲内です。最大{primary_risk['最大枚数']}枚、"
                f"想定利益は{money(primary_risk['最大想定利益'])}です。"
            )

        st.subheader("発注前の確認")
        check1 = st.checkbox("5分足の終値がエントリーラインを明確に抜けた")
        check2 = st.checkbox("エントリー時に損切り注文を同時に置く")
        check3 = st.checkbox("今日の損失上限までの残り枠を確認した")
        check4 = st.checkbox("取り返すための取引ではない")

        if all([check1, check2, check3, check4]) and not status_messages:
            st.success("チェック完了：ルール上は発注候補です。")
        elif not all([check1, check2, check3, check4]):
            st.info("4項目がそろうまで発注しません。")

        with st.expander(secondary_label):
            secondary_risk = plan_risk(
                secondary_plan,
                account_balance,
                risk_percent,
                daily_limit_percent,
                today_pnl,
            )
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("エントリー", f"{secondary_plan['エントリー']:,}円")
            r2.metric("損切り", f"{secondary_plan['損切り']:,}円")
            r3.metric("第一利確", f"{secondary_plan['第一利確']:,}円")
            r4.metric("許容枚数", f"{secondary_risk['最大枚数']}枚")

        st.caption(
            "この計画は入力した直近高値・安値を基にした機械的なラインです。"
            "経済指標、急なニュース、寄り付き直後の急変動では見送る判断を優先します。"
        )

with tab3:
    st.subheader("過去10年の週方向・簡易検証")
    st.caption("判定が出た日の終値から5営業日後までの変動を集計しています。手数料・スリッページ・夜間ギャップ・損切りは未反映です。")

    rows = []
    for label, stats in [("買い優勢", buy_stats), ("売り優勢", sell_stats)]:
        if stats:
            rows.append(
                {
                    "判定": label,
                    "該当回数": stats["回数"],
                    "勝率": f"{stats['勝率']:.1f}%",
                    "平均変動率": percent(stats["平均変動率"]),
                    "中央値": percent(stats["中央値"]),
                    "最大不利変動": percent(stats["最大不利変動"]),
                    "最大連敗": f"{stats['最大連敗']}回",
                }
            )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "次の段階では、日経225マイクロ先物の5分足CSVを読み込み、"
        "上抜け・下抜け・VWAP・損切り・利確まで含めた実戦型バックテストを追加します。"
    )

with tab4:
    st.subheader("このアプリの固定ルール")
    st.write("1. 週方向が見送りなら、5分足ブレイクが出るまで待つ。")
    st.write("2. 損切り幅が上限を超える場合は、チャンスに見えても見送る。")
    st.write("3. 損切り注文はエントリーと同時に置く。")
    st.write("4. 1日の損失上限に達したら、その日は終了する。")
    st.write("5. 連敗後にロットを増やさない。")
    st.write("6. 実際の優位性は、5分足データでのバックテスト結果を優先する。")

    st.warning(
        "このアプリは売買助言や利益を保証するものではありません。"
        "期待値が確認できたルールだけを、小さな枚数から運用してください。"
    )
