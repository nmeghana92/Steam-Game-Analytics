import os
import ast
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from groq import Groq
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

CSV_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "games_march2025_cleaned.csv"
)

@st.cache_data
def fetch_game_data():
    dataset = pd.read_csv(
        CSV_FILE,
        engine="python",
        on_bad_lines="skip"
    )

    essential_cols = [
        "appid", "name", "genres", "price",
        "positive", "negative"
    ]
    
    dataset = dataset[essential_cols].copy()
    dataset = dataset.dropna(subset=["name"])
    dataset = dataset.drop_duplicates(subset=["appid"])
    dataset = dataset.reset_index(drop=True)

    def extract_genres(val):
        if pd.isna(val):
            return []
        try:
            parsed_val = ast.literal_eval(val)
            return parsed_val if isinstance(parsed_val, list) else []
        except (ValueError, SyntaxError, TypeError):
            return []

    dataset["genre_list"] = dataset["genres"].apply(extract_genres)

    for num_col in ["price", "positive", "negative"]:
        dataset[num_col] = pd.to_numeric(
            dataset[num_col], errors="coerce"
        ).fillna(0)

    binarizer = MultiLabelBinarizer()
    genre_features = binarizer.fit_transform(dataset["genre_list"])

    return dataset, genre_features

def get_similar_games(target_game, max_results=5):
    game_df, genre_features = fetch_game_data()

    found_games = game_df[
        game_df["name"].str.casefold() == target_game.strip().casefold()
    ]

    if found_games.empty:
        return pd.DataFrame()

    target_idx = found_games.index[0]

    # Calculate individual score components
    genre_sim = cosine_similarity(
        genre_features[target_idx:target_idx + 1],
        genre_features
    ).flatten()

    game_prices = game_df["price"].clip(lower=0)
    target_price = game_prices.iloc[target_idx]

    price_sim = (
        1 - (game_prices - target_price).abs()
        / max(float(game_prices.max()), 1)
    ).clip(0, 1).to_numpy()

    log_popularity = np.log1p(game_df["positive"].clip(lower=0))
    highest_pop = max(float(log_popularity.max()), 1)

    pop_sim = (
        1 - (log_popularity - log_popularity.iloc[target_idx]).abs()
        / highest_pop
    ).clip(0, 1).to_numpy()

    # Final weighted similarity
    total_scores = (
        0.70 * genre_sim
        + 0.15 * price_sim
        + 0.15 * pop_sim
    )

    total_scores[target_idx] = -1  # Exclude the queried game

    best_matches = np.argsort(total_scores)[-max_results:][::-1]

    recommendations = game_df.iloc[best_matches][
        ["name", "price", "genres", "positive", "negative"]
    ].copy()

    recommendations["similarity"] = total_scores[best_matches].round(3)

    return recommendations.reset_index(drop=True)

def generate_ai_explanation(target_game, rec_df):
    if rec_df is None or rec_df.empty:
        return "No recommendations found to explain."

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("API key for GROQ is missing.")

    ai_client = Groq(api_key=groq_key)

    details_str = rec_df[
        ["name", "price", "genres", "positive", "negative", "similarity"]
    ].to_string(index=False)

    sys_prompt = f"""
You are a helpful Steam game recommendation expert.

Target game: {target_game}

Suggestions:
{details_str}

Provide a simple English explanation of why these games were chosen.
Highlight shared genres, pricing, and the similarity scores. 
Note that similarity is a custom ranking metric, not a probability.
Stick strictly to the provided data without making up gameplay features.
Keep it beginner-friendly and short.
"""

    reply = ai_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": sys_prompt}],
        reasoning_effort="low",
        max_completion_tokens=1024
    )

    return reply.choices[0].message.content or "Failed to generate an explanation."

def fetch_eda_data():
    import os
    import pandas as pd
    import streamlit as st

    dataset_loc = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "games_march2025_cleaned.csv"
    )

    @st.cache_data
    def read_csv_data(filepath):
        return pd.read_csv(filepath, engine="python", on_bad_lines="skip")

    return read_csv_data(dataset_loc)