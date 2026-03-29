import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Telugu Movie Recommendation", page_icon="🎬", layout="wide")

BACKGROUND_IMAGE_URL = "https://static.toiimg.com/thumb/msid-126069810,imgsize-73295,width-1600,height-900,resizemode-75/ai-image.jpg"

background_css = """
    <style>
    .stApp {
        background:
            linear-gradient(rgba(14, 7, 7, 0.78), rgba(33, 13, 10, 0.86)),
            url('__BACKGROUND_IMAGE_URL__');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f8f1e7;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(rgba(255, 244, 214, 0.05), rgba(255, 244, 214, 0.01)),
            repeating-linear-gradient(
                90deg,
                rgba(255, 220, 150, 0.04) 0,
                rgba(255, 220, 150, 0.04) 2px,
                transparent 2px,
                transparent 140px
            );
        opacity: 0.45;
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(44, 15, 15, 0.92), rgba(20, 8, 8, 0.95));
    }

    [data-testid="stSidebar"] * {
        color: #f6e9da;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 248, 240, 0.08);
        border: 1px solid rgba(255, 214, 153, 0.18);
        border-radius: 18px;
        padding: 0.75rem 0.5rem;
    }

    [data-testid="stSelectbox"] > div,
    [data-testid="stTextInput"] > div,
    [data-testid="stMarkdownContainer"],
    [data-testid="stAlertContainer"],
    .st-emotion-cache-1r6slb0,
    .st-emotion-cache-13k62yr {
        color: #f8f1e7;
    }

    .stButton > button {
        border-radius: 999px;
        border: 1px solid rgba(255, 210, 140, 0.45);
        background: linear-gradient(90deg, #a4341d, #d46a1f);
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
    }

    .stButton > button:hover {
        border-color: rgba(255, 229, 189, 0.7);
        background: linear-gradient(90deg, #b53b22, #e27b29);
        color: white;
    }

    [data-testid="stVerticalBlock"] [data-testid="stContainer"] {
        background: rgba(23, 10, 10, 0.62);
        border: 1px solid rgba(255, 214, 153, 0.12);
        border-radius: 22px;
        backdrop-filter: blur(10px);
    }

    h1, h2, h3 {
        color: #fff4df;
        letter-spacing: 0.02em;
    }

    p, label, .stCaption {
        color: #f2e3cf;
    }
    </style>
"""

st.markdown(
    background_css.replace("__BACKGROUND_IMAGE_URL__", BACKGROUND_IMAGE_URL),
    unsafe_allow_html=True,
)


# Fallback only. The cleaned dataset should provide a language column.
NON_TELUGU_TITLES = {
    "Dhoom:3",
    "Ra.One",
    "Dhoom:2",
    "Krrish 3",
    "War",
    "Theri",
    "Billa",
    "7aum Arivu",
}


def normalize_column_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(".", "_")
        .replace(" ", "_")
    )


def get_text_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name in df.columns:
        return df[column_name].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


@st.cache_data
def load_data() -> pd.DataFrame:
    dataset_path = "TeluguMovies_dataset_cleaned.csv"
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        df = pd.read_csv("TeluguMovies_dataset.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    df.columns = [normalize_column_name(col) for col in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str).str.strip()

    numeric_columns = ["year", "runtime", "rating", "no_of_ratings"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "movie" not in df.columns:
        raise ValueError("The dataset must contain a 'Movie' column.")

    df = df[df["movie"] != ""].drop_duplicates(subset="movie").reset_index(drop=True)
    return df


def prepare_recommender(df: pd.DataFrame):
    working_df = df.copy()
    working_df["tags"] = (
        get_text_series(working_df, "overview") + " " +
        get_text_series(working_df, "genre")
    ).str.strip()

    vectorizer = CountVectorizer(stop_words="english")
    count_matrix = vectorizer.fit_transform(working_df["tags"])
    similarity = cosine_similarity(count_matrix)
    movie_lookup = {
        movie.lower(): idx for idx, movie in enumerate(working_df["movie"])
    }
    return working_df, similarity, movie_lookup


def recommend(movie_name: str, recommender_df: pd.DataFrame, similarity, movie_lookup, limit: int = 5):
    movie_index = movie_lookup.get(movie_name.lower())
    if movie_index is None:
        return pd.DataFrame()

    distances = list(enumerate(similarity[movie_index]))
    ranked = sorted(distances, key=lambda item: item[1], reverse=True)[1:limit + 1]
    indices = [idx for idx, _ in ranked]
    return recommender_df.iloc[indices]


def format_runtime(runtime_value) -> str:
    if pd.isna(runtime_value):
        return "N/A"
    return f"{int(runtime_value)} min"


def format_value(value, suffix: str = "") -> str:
    if pd.isna(value) or value == "":
        return "N/A"
    return f"{value}{suffix}"


movies = load_data()

st.title("🎬 Telugu Movie Recommendation System")
st.caption("Pick a Telugu movie and get similar suggestions from the dataset.")

show_curated_only = st.toggle("Show Telugu-focused titles only", value=True)

display_movies = movies.copy()
if show_curated_only:
    if "language" in display_movies.columns:
        display_movies = display_movies[
            display_movies["language"].astype(str).str.lower().eq("telugu")
        ].reset_index(drop=True)
    else:
        display_movies = display_movies[
            ~display_movies["movie"].isin(NON_TELUGU_TITLES)
        ].reset_index(drop=True)

if display_movies.empty:
    st.warning("No movies are available with the current filter.")
    st.stop()

recommender_df, similarity, movie_lookup = prepare_recommender(display_movies)

search_text = st.text_input("Search movie", placeholder="Type a Telugu movie name...")
movie_options = recommender_df["movie"].tolist()

if search_text:
    filtered_options = [movie for movie in movie_options if search_text.lower() in movie.lower()]
    if filtered_options:
        movie_options = filtered_options

selected_movie = st.selectbox("Select movie", movie_options)

if st.button("Show Recommendations", type="primary"):
    recommendations = recommend(selected_movie, recommender_df, similarity, movie_lookup)

    if recommendations.empty:
        st.error("Could not find recommendations for the selected movie.")
    else:
        st.subheader(f"Because you liked {selected_movie}")

        for _, movie_data in recommendations.iterrows():
            with st.container(border=True):
                st.markdown(f"### {movie_data['movie']}")

                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                info_col1.metric("Year", format_value(movie_data.get("year")))
                info_col2.metric("Rating", format_value(movie_data.get("rating")))
                info_col3.metric("Runtime", format_runtime(movie_data.get("runtime")))
                info_col4.metric("Votes", format_value(movie_data.get("no_of_ratings")))

                st.write(f"**Certificate:** {format_value(movie_data.get('certificate'))}")
                st.write(f"**Genre:** {format_value(movie_data.get('genre'))}")
                st.write(f"**Overview:** {format_value(movie_data.get('overview'))}")
