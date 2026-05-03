import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="LLM Hallucination Tracker", page_icon="🧠", layout="wide")
st.title("LLM Hallucination & Bias Live-Tracker")
st.caption("Auto-updates daily. Check which LLM is most trustworthy!")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker_v2.db")

try:
    if not os.path.exists(DB_PATH):
        st.warning("Database not found. Please run the main pipeline first (`python main.py`).")
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM evaluations ORDER BY date DESC", conn)

        if df.empty:
            st.warning("No evaluations found yet. Run the main pipeline to gather data.")
        else:
            # Add Category Filter
            categories = ["All"] + df["category"].unique().tolist()
            selected_category = st.selectbox("Select News Category:", categories)
            
            if selected_category != "All":
                df = df[df["category"] == selected_category]

            if df.empty:
                st.warning("No data for this category.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Evaluations", len(df))
                col2.metric("Most Hallucinating Model", df.groupby("requested_model")["hallucination_detected"].mean().idxmax())
                col3.metric("Most Biased Model", df.groupby("requested_model")["bias_detected"].mean().idxmax())

                # Hallucination rate over time per model
                fig = px.line(df, x="date", y="hallucination_detected", color="requested_model",
                              title=f"Hallucination Rate Over Time ({selected_category})")
                st.plotly_chart(fig, use_container_width=True)

                # Factuality leaderboard
                fig2 = px.bar(df.groupby("requested_model")["factuality_score"].mean().reset_index(),
                              x="requested_model", y="factuality_score", title="Factuality Score Leaderboard",
                              color="factuality_score", color_continuous_scale="RdYlGn")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Show Bias Breakdown
                st.subheader("Bias Detection Breakdown")
                bias_df = df[df["bias_detected"] == 1].groupby(["requested_model", "bias_type"]).size().reset_index(name="count")
                if not bias_df.empty:
                    fig3 = px.bar(bias_df, x="requested_model", y="count", color="bias_type", title="Types of Bias Detected by Model")
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No bias detected yet in this dataset.")
                    
                # Raw Data View
                with st.expander("View Raw Data & Exact Model Versions"):
                    st.dataframe(df[["date", "category", "requested_model", "exact_model", "factuality_score", "hallucination_detected", "overconfidence_score", "bias_detected"]])

except Exception as e:
    st.error(f"Error loading dashboard: {e}. Please ensure the pipeline has been run successfully.")
