import pandas as pd
import streamlit as st
import plotly.express as px

# Page configuration
st.set_page_config(page_title="COE Analytics Dashboard", layout="wide")

# Load cleaned dataset
df = pd.read_csv("data/cleaned_coe_initiatives.csv")

# Ensure correct data types
df["Start Date"] = pd.to_datetime(df["Start Date"])
df["End Date"] = pd.to_datetime(df["End Date"])

# Recalculate KPI Achievement % if needed
df["KPI_Achievement_%"] = (df["KPI Achieved"] / df["KPI Target"]) * 100

# Metrics
total_initiatives = len(df)
avg_kpi = df["KPI_Achievement_%"].mean()
completed_count = (df["Status"] == "Completed").sum()
on_track_pct = (df["Status"] == "On Track").sum() / total_initiatives * 100
delayed_pct = (df["Status"] == "Delayed").sum() / total_initiatives * 100

# Aggregations
status_counts = df["Status"].value_counts().reset_index()
status_counts.columns = ["Status", "Count"]

top3 = df.sort_values("Business Benefit", ascending=False).head(3)

# Title
st.title("COE Analytics Mini Dashboard")
st.markdown("Simulated dashboard for enterprise initiatives tracking")

# KPI Cards


kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric("Total Initiatives", total_initiatives)

with kpi2:
    st.metric("Avg KPI Achievement %", f"{avg_kpi:.2f}%")

with kpi3:
    st.metric("Completed Initiatives", completed_count)

#Status cards

status_col1, status_col2 = st.columns(2)

with status_col1:
    st.metric("On Track Initiatives", f"{on_track_pct:.1f}%")
    st.progress(on_track_pct/100)

with status_col2:
    st.metric("Delayed Initiatives", f"{delayed_pct:.1f}%")
    st.progress(delayed_pct/100)



# Charts Row
col4, col5 = st.columns(2)

with col4:
    st.subheader("Initiative Status Distribution")
    fig_pie = px.pie(
        status_counts,
        names="Status",
        values="Count",
        hole=0.4
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col5:
    st.subheader("Top 3 Initiatives by Business Benefit")
    fig_bar = px.bar(
        top3,
        x="Initiative Name",
        y="Business Benefit",
        text="Business Benefit"
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(yaxis_tickprefix="₹")
    st.plotly_chart(fig_bar, use_container_width=True)

# Detailed table
st.subheader("Initiatives Overview")
display_df = df.copy()
display_df["Start Date"] = display_df["Start Date"].dt.date
display_df["End Date"] = display_df["End Date"].dt.date
display_df["Business Benefit"] = display_df["Business Benefit"].apply(lambda x: f"₹{x:,.0f}")
display_df["KPI_Achievement_%"] = display_df["KPI_Achievement_%"].round(2)

st.dataframe(
    display_df[
        [
            "Initiative Name",
            "Owner",
            "Start Date",
            "End Date",
            "Status",
            "KPI Target",
            "KPI Achieved",
            "KPI_Achievement_%",
            "Business Benefit"
        ]
    ],
    use_container_width=True
)

# Quick insights
st.subheader("Quick Insights")
st.write(f"- Total initiatives tracked: **{total_initiatives}**")
st.write(f"- Average KPI achievement: **{avg_kpi:.2f}%**")
st.write(f"- On Track initiatives: **{on_track_pct:.1f}%**")
st.write(f"- Delayed initiatives: **{delayed_pct:.1f}%**")
st.write(f"- Highest business benefit initiative: **{top3.iloc[0]['Initiative Name']}**")