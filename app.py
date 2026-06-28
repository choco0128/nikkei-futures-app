import streamlit as st

st.set_page_config(
    page_title="日経225先物 判定",
    page_icon="📈",
    layout="wide",
)

st.title("日経225先物｜方向判定アプリ")
st.caption("週・次セッションの買い優勢／売り優勢／見送りを、過去データから判定するアプリ")

st.info(
    "アプリの土台を作成しました。"
    "現在はデータ未接続のため、売買判定はまだ行いません。"
)

st.divider()

st.subheader("現在の判定")
st.warning("データ未接続のため、まだ判定できません。")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("週方向", "準備中")

with col2:
    st.metric("次セッション", "準備中")

with col3:
    st.metric("信頼度", "-- 点")

st.divider()

st.subheader("今後追加する機能")
st.write("・週足からの買い優勢／売り優勢／見送り判定")
st.write("・次セッションのエントリー条件、損切り、利確候補")
st.write("・過去の類似局面の勝率、平均損益、最大連敗")
st.write("・資金額と損切り幅からの許容ロット計算")
