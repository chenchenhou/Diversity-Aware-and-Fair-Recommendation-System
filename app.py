import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loading import load_data, preprocess_data, split_data
from svd_recommender import train_svd_recommender, generate_svd_candidates
from evaluation import evaluate_svd_recs

# Set page config
st.set_page_config(
    page_title="Diversity-Aware Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Load and cache data
@st.cache_data
def load_all_data():
    ratings_df, users_df, movies_df = load_data("ml-1m/ratings.dat", "ml-1m/users.dat", "ml-1m/movies.dat")
    ratings_df, users_df, movies_df = preprocess_data(ratings_df, users_df, movies_df)
    train_df, test_df = split_data(ratings_df, test_ratio=0.2)
    return ratings_df, users_df, movies_df, train_df, test_df

# Load data
ratings_df, users_df, movies_df, train_df, test_df = load_all_data()

# Train model
@st.cache_resource
def get_model():
    model, trainset, item_factors = train_svd_recommender(train_df)
    return model, trainset, item_factors

model, trainset, item_factors = get_model()

# Initialize session state for parameter storage
if 'w_rel' not in st.session_state:
    st.session_state.w_rel = 0.6
if 'w_div' not in st.session_state:
    st.session_state.w_div = 0.2
if 'w_fair' not in st.session_state:
    st.session_state.w_fair = 0.2
if 'candidate_pool_size' not in st.session_state:
    st.session_state.candidate_pool_size = 20
if 'N' not in st.session_state:
    st.session_state.N = 5

# Add a session key to force recalculation only when parameters are applied
if 'session_id' not in st.session_state:
    st.session_state.session_id = 0

# Sidebar for user selection and parameters
st.sidebar.title("📋 Controls")
user_id = st.sidebar.selectbox(
    "Select a user",
    options=users_df['UserID'].unique(),
    format_func=lambda x: f"User {x}",
    key="user_selector"
)

# If user changes, we should update recommendations
if 'previous_user' not in st.session_state:
    st.session_state.previous_user = user_id
if st.session_state.previous_user != user_id:
    st.session_state.previous_user = user_id
    st.session_state.session_id += 1  # Force recalculation

st.sidebar.markdown("---")
st.sidebar.subheader("Recommendation Parameters")
st.sidebar.markdown("Adjust the weights to balance between different objectives:")

# Create a form to capture all parameter changes
with st.sidebar.form("parameter_form"):
    # Add sliders for the recommendation weights
    w_rel = st.slider("Relevance Weight", 0.0, 1.0, st.session_state.w_rel, 0.1, 
                     help="How much to prioritize matching user preferences")
    w_div = st.slider("Diversity Weight", 0.0, 1.0, st.session_state.w_div, 0.1,
                     help="How much to prioritize recommendation variety")
    w_fair = st.slider("Fairness Weight", 0.0, 1.0, st.session_state.w_fair, 0.1,
                      help="How much to prioritize fair representation of genres")

    # Candidate pool size
    candidate_pool_size = st.slider("Candidate Pool Size", 10, 50, st.session_state.candidate_pool_size, 5,
                                   help="Number of initial candidates to consider before reranking")

    # Number of recommendations
    N = st.slider("Number of Recommendations", 3, 10, st.session_state.N, 1,
                 help="Number of movies to recommend")
    
    # Display the normalized weights preview
    total = w_rel + w_div + w_fair
    norm_w_rel, norm_w_div, norm_w_fair = w_rel/total, w_div/total, w_fair/total
    st.markdown(f"*Preview of normalized weights:*")
    st.markdown(f"*Relevance ({norm_w_rel:.2f}), Diversity ({norm_w_div:.2f}), Fairness ({norm_w_fair:.2f})*")
    
    # Add a submit button to apply changes
    submit_button = st.form_submit_button("Apply Parameters")
    
    if submit_button:
        # Ensure weights sum to 1.0
        total = w_rel + w_div + w_fair
        st.session_state.w_rel = w_rel / total
        st.session_state.w_div = w_div / total
        st.session_state.w_fair = w_fair / total
        st.session_state.candidate_pool_size = candidate_pool_size
        st.session_state.N = N
        st.session_state.session_id += 1  # Increment the session ID to invalidate cache

# Use the session state values for actual recommendations
w_rel, w_div, w_fair = st.session_state.w_rel, st.session_state.w_div, st.session_state.w_fair
candidate_pool_size = st.session_state.candidate_pool_size
N = st.session_state.N

# Main content
st.title("🎬 Diversity-Aware Movie Recommender")
st.markdown("""
This demo showcases a movie recommendation system that balances multiple objectives:
- **Relevance**: How well the recommendations match user preferences
- **Diversity**: How varied the recommendations are
- **Fairness**: How fairly different movie categories are represented

Adjust the parameters in the sidebar and click "Apply Parameters" to update recommendations.
""")

# Get recommendations with caching
@st.cache_data(show_spinner=False)
def get_recommendations(user_id, N=5, candidate_pool_size=20, w_rel=0.6, w_div=0.2, w_fair=0.2, _session_id=None):
    # This _session_id parameter helps ensure the cache is invalidated properly
    # Get regular SVD recommendations
    regular_candidates = generate_svd_candidates(model, trainset, user_id, movies_df, candidate_pool_size)
    regular_recs = [mid for mid, _ in regular_candidates[:N]]
    regular_scores = [score for _, score in regular_candidates[:N]]
    
    # Get diversity-aware recommendations
    diversity_recs, diversity_rel, diversity_div, diversity_fair = evaluate_svd_recs(
        model,
        trainset,
        item_factors,
        test_df,
        users_df,
        movies_df,
        N=N,
        candidate_pool_size=candidate_pool_size,
        w_rel=w_rel,
        w_div=w_div,
        w_fair=w_fair,
        return_recs=True,
        user_id=user_id,
        return_metrics=True
    )
    
    return regular_recs, regular_scores, diversity_recs, diversity_rel, diversity_div, diversity_fair

# Load recommendations
with st.spinner("Generating recommendations..."):
    regular_recs, regular_scores, diversity_recs, diversity_rel, diversity_div, diversity_fair = get_recommendations(
        user_id, N, candidate_pool_size, w_rel, w_div, w_fair, _session_id=st.session_state.session_id
    )

# Recommendation Metrics
st.header("📈 Recommendation Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Relevance Score",
        f"{diversity_rel:.2f}",
        delta=f"{diversity_rel - np.mean(regular_scores):.2f}",
        help="How well the recommendations match user preferences"
    )

with col2:
    st.metric(
        "Diversity Score",
        f"{diversity_div:.2f}",
        help="How varied the recommendations are"
    )

with col3:
    st.metric(
        "Fairness Score",
        f"{diversity_fair:.2f}",
        help="How fairly different movie categories are represented"
    )

# Display current applied parameters
st.markdown(f"*Currently applied parameters: Relevance ({w_rel:.2f}), Diversity ({w_div:.2f}), Fairness ({w_fair:.2f})*")

# Display recommendations
st.header("🎞️ Movie Recommendations")
col1, col2 = st.columns(2)

# Function to display movie
def display_movie(movie):
    title = movie['Title']
    # Extract year from title if present, otherwise use empty string
    year = ""
    if '(' in title and ')' in title:
        year_start = title.rfind('(')
        year_end = title.rfind(')')
        if year_start < year_end:
            year = title[year_start:year_end+1]
            title = title[:year_start].strip()
    
    st.markdown(f"**{title}** {year}")
    st.markdown(f"*Genres: {', '.join(movie['Genres']) if isinstance(movie['Genres'], list) else movie['Genres']}*")

with col1:
    st.subheader("Standard Recommendations")
    st.markdown("*Based purely on predicted user preference*")
    for i, (movie_id, _) in enumerate(zip(regular_recs, regular_scores), 1):
        with st.container():
            movie = movies_df[movies_df['MovieID'] == movie_id].iloc[0]
            display_movie(movie)
            st.markdown("---")

with col2:
    st.subheader("Diversity-Aware Recommendations")
    st.markdown("*Balanced for relevance, diversity, and fairness*")
    for i, movie_id in enumerate(diversity_recs, 1):
        with st.container():
            movie = movies_df[movies_df['MovieID'] == movie_id].iloc[0]
            display_movie(movie)
            st.markdown("---")

# Visualization section
st.header("📊 Recommendation Analysis")

# Genre distribution comparison
tab1, tab2, tab3 = st.tabs(["Genre Distribution", "User Preferences", "Parameter Impact"])

with tab1:
    # Create a more detailed genre distribution visualization
    regular_genres = movies_df[movies_df['MovieID'].isin(regular_recs)]['Genres'].explode()
    diversity_genres = movies_df[movies_df['MovieID'].isin(diversity_recs)]['Genres'].explode()
    
    # Combine and get counts
    all_genres = pd.concat([
        pd.DataFrame({'Genre': regular_genres, 'Method': 'Standard'}),
        pd.DataFrame({'Genre': diversity_genres, 'Method': 'Diversity-Aware'})
    ])
    
    fig = px.histogram(
        all_genres, 
        x='Genre', 
        color='Method',
        barmode='group',
        title='Genre Distribution Comparison',
        color_discrete_map={
            'Standard': '#1f77b4',
            'Diversity-Aware': '#ff7f0e'
        }
    )
    
    fig.update_layout(
        xaxis_title='Genre',
        yaxis_title='Count',
        legend_title='Recommendation Method'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    This chart shows how genres are distributed in both recommendation approaches. 
    The diversity-aware approach aims for a more balanced representation across genres.
    """)

with tab2:
    # User preferences visualization
    user_ratings = ratings_df[ratings_df['UserID'] == user_id]
    rated_movies = movies_df[movies_df['MovieID'].isin(user_ratings['MovieID'])]
    
    # Join to get ratings
    rated_movies = rated_movies.merge(
        user_ratings[['MovieID', 'Rating']], 
        on='MovieID', 
        how='left'
    )
    
    # Create genre-rating pairs
    genre_ratings = []
    for _, movie in rated_movies.iterrows():
        genres = movie['Genres'] if isinstance(movie['Genres'], list) else [movie['Genres']]
        for genre in genres:
            genre_ratings.append({
                'Genre': genre,
                'Rating': movie['Rating'],
                'Movie': movie['Title']
            })
    
    genre_df = pd.DataFrame(genre_ratings)
    
    # Calculate average rating per genre
    genre_avg = genre_df.groupby('Genre')['Rating'].mean().reset_index()
    genre_count = genre_df.groupby('Genre').size().reset_index(name='Count')
    genre_stats = genre_avg.merge(genre_count, on='Genre')
    
    # Create a bubble chart
    fig = px.scatter(
        genre_stats,
        x='Genre',
        y='Rating',
        size='Count',
        title=f'User {user_id} Genre Preferences',
        color='Rating',
        color_continuous_scale='viridis',
        size_max=50
    )
    
    fig.update_layout(
        xaxis_title='Genre',
        yaxis_title='Average Rating',
        coloraxis_colorbar_title='Avg Rating'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    This visualization shows your genre preferences based on past ratings:
    - Bubble size represents how many movies of each genre you've rated
    - Position on the y-axis shows your average rating for that genre
    """)

with tab3:
    # Create placeholder for parameter impact visualization
    st.markdown("""
    ### How Parameters Affect Recommendations
    
    The sliders in the sidebar let you control the balance between three objectives:
    
    **Relevance (Higher = More Personalized)**  
    Prioritizes movies that closely match your past preferences. Higher weights lead to recommendations that are similar to movies you've enjoyed before.
    
    **Diversity (Higher = More Variety)**  
    Encourages variety in the recommendations. Higher weights lead to recommendations that cover a broader range of genres and styles.
    
    **Fairness (Higher = More Balanced)**  
    Ensures different genres receive fair representation. Higher weights lead to recommendations that include genres that might otherwise be underrepresented.
    
    *Adjust the sliders and click "Apply Parameters" to see how the recommendations change!*
    """)
    
    # Create a simple radar chart showing current parameter balance
    params = ['Relevance', 'Diversity', 'Fairness']
    values = [w_rel, w_div, w_fair]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=params,
        fill='toself',
        name='Current Settings'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Current Parameter Balance"
    )
    
    st.plotly_chart(fig, use_container_width=True)
