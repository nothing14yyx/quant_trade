import pandas as pd
import altair as alt
import streamlit as st
from quant_trade.monitor import fetch_recent
from quant_trade.utils import load_config, connect_mysql
from quant_trade.utils.db import CONFIG_PATH


def radar_chart(data: dict) -> alt.Chart:
    df = pd.DataFrame({"factor": list(data.keys()), "value": list(data.values())})
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(theta="factor", radius="value")
    )


def main():
    cfg = load_config(CONFIG_PATH)
    engine = connect_mysql(cfg)
    sig, fac = fetch_recent(engine, limit=500)

    st.title("Quant Trade Dashboard")
    st.subheader("Score Trend")
    st.line_chart(sig.set_index("open_time")["score"])

    st.subheader("IC Curve")
    ic_df = sig.dropna(subset=["score", "signal"]).copy()
    ic_df["ic"] = ic_df["score"].rolling(50).corr(ic_df["signal"])
    ic_chart = alt.Chart(ic_df).mark_line().encode(x="open_time:T", y="ic:Q")
    st.altair_chart(ic_chart, use_container_width=True)

    if not fac.empty:
        st.subheader("Latest Factor Contribution")
        radar = radar_chart(fac.iloc[-1].drop("open_time").to_dict())
        st.altair_chart(radar, use_container_width=True)


if __name__ == "__main__":
    main()

