import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from recommendation import (
    get_similar_games,
    generate_ai_explanation,
    fetch_eda_data
)
from model_evaluation import (
    run_model_eval,
    analyze_price_features
)

st.set_page_config(
    page_title="Steam Game Insights",
    page_icon="🕹️",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #f7f9fc; }
h1 { color: #1e3a8a; }
div.stButton > button { background-color: #3b82f6; color: white; border-radius: 8px; font-weight: bold; }
div.stButton > button:hover { background-color: #2563eb; color: white; }
div[data-testid="stMetric"] { background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #d1d5db; }
</style>
""", unsafe_allow_html=True)

st.title("🕹️ AI Steam Game Insights & Recommendations")
st.write("Explore game stats, evaluate review sentiments, and find your next favorite game using AI.")
st.divider()

# --- RECOMMENDATION ENGINE ---
st.header("🔍 Discover Similar Games")

left_col, right_col = st.columns([3, 1])

with left_col:
    search_query = st.text_input("Search for a game title:", value="Counter-Strike 2")

with right_col:
    num_results = st.selectbox("Results to show:", [3, 5, 7, 10], index=1)

if st.button("Find Matches 🚀", use_container_width=True):
    if not search_query.strip():
        st.warning("Please provide a valid game name.")
    else:
        with st.spinner("Searching database..."):
            results_df = get_similar_games(search_query, num_results)

        if results_df.empty:
            st.warning("Could not locate that game. Please check your spelling.")
        else:
            st.success(f"Showing top matches for: {search_query}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Matches", len(results_df))
            m2.metric("Mean Similarity", f"{results_df['similarity'].mean():.1%}")
            m3.metric("Free Options", int((results_df["price"] == 0).sum()))

            st.subheader("🎯 Best Matches")
            
            ui_df = results_df.copy()
            ui_df["price"] = ui_df["price"].apply(lambda p: "Free" if p == 0 else f"${p:.2f}")
            ui_df["similarity"] = ui_df["similarity"].apply(lambda s: f"{s:.1%}")
            ui_df = ui_df.rename(columns={
                "name": "Title", "price": "Cost", "genres": "Tags",
                "positive": "Upvotes", "negative": "Downvotes", "similarity": "Match Score"
            })

            st.dataframe(ui_df, use_container_width=True, hide_index=True)

            st.subheader("🧠 Why These Games?")
            try:
                with st.spinner("Asking AI for insights..."):
                    ai_text = generate_ai_explanation(search_query, results_df)
                st.info(ai_text)
            except Exception as e:
                st.error(f"Failed to load AI summary: {e}")

st.divider()

# --- EXPLORATORY DATA ANALYSIS ---
st.header("📊 Global Data Analysis")

try:
    stats_df = fetch_eda_data()
    for col in ["price", "positive", "negative"]:
        if col in stats_df.columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce")

    st.subheader("📌 Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Catalog Size", f"{len(stats_df):,}")

    if "price" in stats_df.columns:
        kpi2.metric("Free to Play", f"{int((stats_df['price'].fillna(0) == 0).sum()):,}")
        kpi3.metric("Paid Titles", f"{int((stats_df['price'].fillna(0) > 0).sum()):,}")
    
    if "name" in stats_df.columns:
        kpi4.metric("Distinct Titles", f"{stats_df['name'].nunique():,}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💰 Monetization Model")
        if "price" in stats_df.columns:
            pricing = stats_df["price"].fillna(0).apply(lambda p: "Free" if p == 0 else "Paid")
            st.bar_chart(pricing.value_counts().reindex(["Free", "Paid"], fill_value=0))

    with c2:
        st.subheader("⭐ Most Upvoted")
        if {"name", "positive"}.issubset(stats_df.columns):
            top_reviewed = stats_df[["name", "positive"]].dropna().nlargest(10, "positive").set_index("name")
            st.bar_chart(top_reviewed)

except Exception as e:
    st.error(f"Error loading EDA: {e}")

st.divider()

# --- ML EVALUATION ---
st.header("🤖 Sentiment Prediction Models")
st.write("Evaluating classifier performance on review sentiments (Positive, Mixed, Negative).")

if st.button("Evaluate Models 📉"):
    try:
        with st.spinner("Training models..."):
            metrics_df, rep_dicts, conf_mats = run_model_eval()

        st.success("Training complete!")
        st.dataframe(metrics_df.style.format({
            "Accuracy": "{:.2%}", "Macro Precision": "{:.2%}", 
            "Macro Recall": "{:.2%}", "Macro F1": "{:.2%}"
        }), use_container_width=True, hide_index=True)

        st.bar_chart(metrics_df.set_index("Algorithm")[["Accuracy"]])

        chosen_alg = st.selectbox("View details for:", metrics_df["Algorithm"].tolist())
        
        st.subheader(f"Confusion Matrix: {chosen_alg}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(conf_mats[chosen_alg], cmap="Purples")
        
        lbls = ["Negative", "Mixed", "Positive"]
        ax.set_xticks(range(3), labels=lbls)
        ax.set_yticks(range(3), labels=lbls)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(conf_mats[chosen_alg][i, j]), ha="center", va="center")
        
        st.pyplot(fig)
        plt.close(fig)

        # --- 1. CLASSIFICATION REPORT & BEST MODEL (Inside the try block) ---
        st.subheader("Classification Report")
        report_df = pd.DataFrame(rep_dicts[chosen_alg]).transpose()
        st.dataframe(report_df, use_container_width=True)

        best_model_idx = metrics_df["Macro F1"].idxmax()
        best_algo = metrics_df.loc[best_model_idx, "Algorithm"]
        best_score = metrics_df.loc[best_model_idx, "Macro F1"]
        st.info(f"Best model by macro F1: {best_algo} ({best_score:.2%})")

    except Exception as e:
        st.error(f"Model error: {e}")

# --- 2. GAME PRICE BY REVIEW SENTIMENT (Outside the model button block) ---
st.header("🔍 Game Price by Review Sentiment")
st.write("Compare game prices across sentiment categories. This is descriptive analysis, not model feature importance.")

try:
    price_stats_df = analyze_price_features()
    
    # Rename columns to match the video execution
    ui_price_df = price_stats_df.rename(columns={
        "Total Games": "Game Count", 
        "Mean Price": "Average Price"
    })
    
    st.dataframe(
        ui_price_df.style.format({
            "Average Price": "${:.2f}", 
            "Median Price": "${:.2f}"
        }), 
        use_container_width=True, 
        hide_index=True
    )

    st.subheader("Average Price by Sentiment")
    st.bar_chart(ui_price_df.set_index("Sentiment")["Average Price"])

except Exception as e:
    st.error(f"Error loading price data: {e}")

st.divider()
st.caption("Powered by Machine Learning & Groq LLM")