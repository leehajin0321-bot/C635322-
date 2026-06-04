import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dataset import df_songs, get_recommendations_by_song, get_recommendations_by_filters

# Set page configuration
st.set_page_config(
    page_title="MelodicMatch - Song Recommendation App",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphism theme and custom fonts
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Main body background */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d1117;
        color: #f0f6fc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(22, 27, 34, 0.95) !important;
        border-right: 1px solid rgba(240, 246, 252, 0.1);
        backdrop-filter: blur(12px);
    }
    
    /* Custom headers and title */
    .hero-container {
        background: linear-gradient(135deg, #1f1a3a 0%, #3b1c55 50%, #5c1d68 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 2.5rem;
        border-radius: 1.5rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(92, 29, 104, 0.3);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a855f7, #f43f5e, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #8b949e;
        font-weight: 400;
    }
    
    /* Section dividers */
    hr {
        border-color: rgba(240, 246, 252, 0.1);
    }
    
    /* Custom card layout classes */
    .card-container {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 1rem;
        padding: 1.25rem;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .card-container:hover {
        transform: translateY(-5px);
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 10px 20px rgba(168, 85, 247, 0.15);
        background: rgba(30, 36, 45, 0.8);
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .card-artist {
        font-size: 1rem;
        color: #a855f7;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .card-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .card-badge-mood {
        display: inline-block;
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }

    /* Playlist summary metric cards */
    .metric-card {
        background: rgba(33, 38, 45, 0.8);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Streamlit widgets adjustment */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1.5rem;
        color: #8b949e;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(30, 36, 45, 0.8);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f1a3a !important;
        border-color: rgba(168, 85, 247, 0.4) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if 'playlist' not in st.session_state:
    st.session_state.playlist = []

def add_to_playlist(song):
    """Callback to add a song to the temporary playlist."""
    if song['title'] not in [s['title'] for s in st.session_state.playlist]:
        # Convert Series/dict to standard dict to avoid type issues
        song_dict = {
            'title': song['title'],
            'artist': song['artist'],
            'genre': song['genre'],
            'release_year': int(song['release_year']),
            'mood': song['mood'],
            'energy': float(song['energy']),
            'valence': float(song['valence']),
            'danceability': float(song['danceability']),
            'acousticness': float(song['acousticness']),
            'tempo_bpm': int(song['tempo_bpm'])
        }
        st.session_state.playlist.append(song_dict)
        st.toast(f"🎵 **{song['title']}** added to your playlist!", icon="✅")
    else:
        st.toast(f"💡 **{song['title']}** is already in your playlist!", icon="ℹ️")

def remove_from_playlist(title):
    """Callback to remove a song from the playlist."""
    st.session_state.playlist = [s for s in st.session_state.playlist if s['title'] != title]
    st.toast(f"🗑️ Removed song from playlist.", icon="ℹ️")

def clear_playlist():
    """Callback to clear the playlist."""
    st.session_state.playlist = []
    st.toast("🗑️ Playlist cleared.", icon="ℹ️")

# Create a radar chart helper function
def create_radar_chart(target_song, recs_df=None):
    categories = ['Energy ⚡', 'Valence 😊', 'Danceability 🕺', 'Acousticness 🎸', 'Tempo 🥁']
    
    fig = go.Figure()
    
    # Target song values
    target_vals = [
        target_song['energy'],
        target_song['valence'],
        target_song['danceability'],
        target_song['acousticness'],
        target_song['normalized_tempo']
    ]
    
    # Add target song trace
    fig.add_trace(go.Scatterpolar(
        r=target_vals,
        theta=categories,
        fill='toself',
        name=f"Selected: {target_song['title']}",
        line_color='#a855f7',
        fillcolor='rgba(168, 85, 247, 0.2)'
    ))
    
    # Add average of recommendations trace if provided
    if recs_df is not None and not recs_df.empty:
        avg_vals = [
            recs_df['energy'].mean(),
            recs_df['valence'].mean(),
            recs_df['danceability'].mean(),
            recs_df['acousticness'].mean(),
            recs_df['normalized_tempo'].mean()
        ]
        fig.add_trace(go.Scatterpolar(
            r=avg_vals,
            theta=categories,
            fill='toself',
            name="Recommendations Avg",
            line_color='#00e1d9',
            fillcolor='rgba(0, 225, 217, 0.15)'
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='rgba(240, 246, 252, 0.15)',
                linecolor='rgba(240, 246, 252, 0.15)',
                tickfont=dict(color='#8b949e')
            ),
            angularaxis=dict(
                gridcolor='rgba(240, 246, 252, 0.15)',
                tickfont=dict(color='#ffffff', size=11)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color='#f0f6fc')
        ),
        margin=dict(l=40, r=40, t=20, b=40)
    )
    return fig

# --- HERO HEADER ---
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">🎵 MelodicMatch</h1>
        <p class="hero-subtitle">Discover tracks tailored to your mood, style, and musical frequency.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("## ⚙️ App Controls")
st.sidebar.info("Adjust configurations below to discover new tracks or explore our 100-song curated catalog.")

# Quick stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 Current Playlist ({len(st.session_state.playlist)} songs)")
if st.session_state.playlist:
    for s in st.session_state.playlist[:5]:
        st.sidebar.markdown(f"- **{s['title']}** - {s['artist']}")
    if len(st.session_state.playlist) > 5:
        st.sidebar.markdown(f"*And {len(st.session_state.playlist) - 5} more...*")
else:
    st.sidebar.markdown("Your playlist is currently empty.")

# Create the Tabs
tab_similar, tab_explorer, tab_playlist = st.tabs([
    "🔍 Similar Song Finder", 
    "🧭 Mood & Genre Explorer", 
    "⭐ My Playlist & Analytics"
])

# --- TAB 1: SIMILAR SONG FINDER ---
with tab_similar:
    st.markdown("### Find Songs Similar to Your Favorite Track")
    st.write("Our content-based recommendation engine calculates the **cosine similarity** between songs using audio features: *Energy, Happiness (Valence), Danceability, Acousticness, and Tempo*.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Select Target Song")
        # Combobox to search/select song
        song_options = sorted(df_songs.apply(lambda r: f"{r['title']} - {r['artist']}", axis=1).tolist())
        selected_option = st.selectbox(
            "Search or select a song:",
            options=song_options,
            index=song_options.index("Ditto - NewJeans") if "Ditto - NewJeans" in song_options else 0
        )
        
        target_title = selected_option.split(" - ")[0]
        target_song = df_songs[df_songs['title'] == target_title].iloc[0]
        
        # Display selected song info
        st.markdown("---")
        st.markdown("##### Selected Song Details")
        st.image(target_song['unsplash_url'], caption=f"{target_song['title']} Album Art", use_container_width=True)
        st.write(f"**Artist:** {target_song['artist']}")
        st.write(f"**Genre:** {target_song['genre']} | **Year:** {target_song['release_year']}")
        st.write(f"**Mood:** {target_song['mood']} | **Tempo:** {target_song['tempo_bpm']} BPM")
        
        # Target song values
        st.write("**Audio Features:**")
        st.progress(float(target_song['energy']), text=f"⚡ Energy: {int(target_song['energy']*100)}%")
        st.progress(float(target_song['valence']), text=f"😊 Happiness (Valence): {int(target_song['valence']*100)}%")
        st.progress(float(target_song['danceability']), text=f"🕺 Danceability: {int(target_song['danceability']*100)}%")
        st.progress(float(target_song['acousticness']), text=f"🎸 Acousticness: {int(target_song['acousticness']*100)}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.button("➕ Add to Playlist", key="add_target", on_click=add_to_playlist, args=(target_song,), use_container_width=True)
        with col_t2:
            yt_url = f"https://www.youtube.com/results?search_query={target_song['youtube_query'].replace(' ', '+')}"
            st.link_button("🎵 Listen on YT", yt_url, use_container_width=True)
            
        st.markdown("---")
        num_recs = st.slider("Number of Recommendations:", min_value=3, max_value=9, value=6, step=3)

    with col2:
        st.markdown("#### Audio Feature Signature Comparison")
        recs_df = get_recommendations_by_song(target_title, top_n=num_recs)
        
        if not recs_df.empty:
            # Generate and display Radar Chart
            radar_fig = create_radar_chart(target_song, recs_df)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            st.markdown("#### Recommended Songs")
            
            # Display recommendations in a grid
            cols = st.columns(3)
            for idx, (_, rec_song) in enumerate(recs_df.iterrows()):
                col_idx = idx % 3
                with cols[col_idx]:
                    similarity_percent = int(rec_song['similarity'] * 100)
                    st.markdown(
                        f"""
                        <div class="card-container">
                            <div style="position: relative;">
                                <img src="{rec_song['unsplash_url']}" style="width:100%; border-radius: 0.5rem; aspect-ratio: 1; object-fit: cover; margin-bottom: 0.5rem;">
                                <span style="position: absolute; top: 8px; right: 8px; background: rgba(109, 40, 217, 0.9); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; border: 1px solid rgba(255,255,255,0.2);">
                                    {similarity_percent}% Match
                                </span>
                            </div>
                            <div class="card-title" title="{rec_song['title']}">{rec_song['title']}</div>
                            <div class="card-artist">{rec_song['artist']}</div>
                            <div>
                                <span class="card-badge">{rec_song['genre']}</span>
                                <span class="card-badge-mood">{rec_song['mood']}</span>
                            </div>
                            <div style="font-size:0.8rem; color:#8b949e; margin-bottom:0.8rem;">
                                Year: {rec_song['release_year']} | Tempo: {rec_song['tempo_bpm']} BPM
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Feature progress indicators
                    st.progress(float(rec_song['energy']), text=f"⚡ Energy: {int(rec_song['energy']*100)}%")
                    st.progress(float(rec_song['valence']), text=f"😊 Happiness: {int(rec_song['valence']*100)}%")
                    
                    c_btn1, c_btn2 = st.columns(2)
                    with c_btn1:
                        # Recs are pandas series or row, pass dict to callback
                        st.button("➕ Add", key=f"add_rec_{rec_song['title']}_{idx}", on_click=add_to_playlist, args=(rec_song,), use_container_width=True)
                    with c_btn2:
                        yt_rec_url = f"https://www.youtube.com/results?search_query={rec_song['youtube_query'].replace(' ', '+')}"
                        st.link_button("🎵 Listen", yt_rec_url, use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.error("Error generating recommendations.")

# --- TAB 2: MOOD & GENRE EXPLORER ---
with tab_explorer:
    st.markdown("### Discover Songs by Mood, Genre & Era")
    st.write("Browse the curated catalog by filtering through specific moods, musical styles, release decades, and sorting based on track attributes.")
    
    # Filtering UI
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    with f_col1:
        mood_list = ["All"] + sorted(list(df_songs['mood'].unique()))
        selected_mood = st.selectbox("Select Mood:", mood_list)
        
    with f_col2:
        genre_list = ["All"] + sorted(list(df_songs['genre'].unique()))
        selected_genre = st.selectbox("Select Genre:", genre_list)
        
    with f_col3:
        min_yr = int(df_songs['release_year'].min())
        max_yr = int(df_songs['release_year'].max())
        selected_years = st.slider("Release Year Range:", min_value=1950, max_value=max_yr, value=(1990, max_yr), step=5)
        
    with f_col4:
        sort_options = {
            "None": None,
            "Energy (High → Low)": "energy",
            "Happiness (High → Low)": "valence",
            "Danceability (High → Low)": "danceability",
            "Acousticness (High → Low)": "acousticness",
            "Tempo (Fast → Slow)": "tempo_bpm"
        }
        selected_sort_label = st.selectbox("Sort By Attribute:", list(sort_options.keys()))
        selected_sort_val = sort_options[selected_sort_label]
        
    st.markdown("---")
    
    # Get filtered songs
    explorer_recs = get_recommendations_by_filters(
        mood=selected_mood,
        genre=selected_genre,
        start_year=selected_years[0],
        end_year=selected_years[1],
        sort_by=selected_sort_val
    )
    
    st.markdown(f"#### Matches Found: **{len(explorer_recs)}** tracks")
    
    if not explorer_recs.empty:
        # Display in a grid of 4 columns
        grid_cols = st.columns(4)
        for idx, (_, song) in enumerate(explorer_recs.iterrows()):
            col_idx = idx % 4
            with grid_cols[col_idx]:
                st.markdown(
                    f"""
                    <div class="card-container">
                        <img src="{song['unsplash_url']}" style="width:100%; border-radius: 0.5rem; aspect-ratio: 1; object-fit: cover; margin-bottom: 0.5rem;">
                        <div class="card-title" title="{song['title']}">{song['title']}</div>
                        <div class="card-artist">{song['artist']}</div>
                        <div>
                            <span class="card-badge">{song['genre']}</span>
                            <span class="card-badge-mood">{song['mood']}</span>
                        </div>
                        <div style="font-size:0.8rem; color:#8b949e; margin-bottom:0.8rem;">
                            Year: {song['release_year']} | Tempo: {song['tempo_bpm']} BPM
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Expandable audio metrics
                with st.expander("⚡ Show Attributes"):
                    st.progress(float(song['energy']), text=f"⚡ Energy: {int(song['energy']*100)}%")
                    st.progress(float(song['valence']), text=f"😊 Happiness: {int(song['valence']*100)}%")
                    st.progress(float(song['danceability']), text=f"🕺 Danceability: {int(song['danceability']*100)}%")
                    st.progress(float(song['acousticness']), text=f"🎸 Acousticness: {int(song['acousticness']*100)}%")
                
                ec_btn1, ec_btn2 = st.columns(2)
                with ec_btn1:
                    st.button("➕ Add", key=f"add_exp_{song['title']}_{idx}", on_click=add_to_playlist, args=(song,), use_container_width=True)
                with ec_btn2:
                    yt_exp_url = f"https://www.youtube.com/results?search_query={song['youtube_query'].replace(' ', '+')}"
                    st.link_button("🎵 Listen", yt_exp_url, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("No tracks match your filter settings. Try expanding your search filters!")

# --- TAB 3: PLAYLIST & ANALYTICS ---
with tab_playlist:
    st.markdown("### Your Custom Playlist & Music Profile Analytics")
    
    if not st.session_state.playlist:
        st.info("Your custom playlist is empty. Browse the tabs above and click 'Add' to build a playlist!")
    else:
        playlist_df = pd.DataFrame(st.session_state.playlist)
        
        # Header controls
        p_hdr1, p_hdr2 = st.columns([3, 1])
        with p_hdr1:
            st.markdown(f"#### Managing **{len(playlist_df)}** tracks")
        with p_hdr2:
            st.button("🗑️ Clear Playlist", on_click=clear_playlist, use_container_width=True, type="secondary")
            
        st.markdown("---")
        
        # Top level metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            avg_energy = int(playlist_df['energy'].mean() * 100)
            energy_emoji = "⚡" if avg_energy > 50 else "🍃"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_energy}%</div>
                <div class="metric-label">{energy_emoji} Avg Energy</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            avg_valence = int(playlist_df['valence'].mean() * 100)
            valence_emoji = "☀️" if avg_valence > 50 else "🌧️"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_valence}%</div>
                <div class="metric-label">{valence_emoji} Avg Happiness</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            avg_dance = int(playlist_df['danceability'].mean() * 100)
            dance_emoji = "🕺" if avg_dance > 50 else "🧘"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_dance}%</div>
                <div class="metric-label">{dance_emoji} Avg Danceability</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            avg_tempo = int(playlist_df['tempo_bpm'].mean())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_tempo}</div>
                <div class="metric-label">🥁 Avg BPM</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts section
        c_col1, c_col2 = st.columns(2)
        
        with c_col1:
            st.markdown("#### Genre Distribution")
            genre_counts = playlist_df['genre'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            
            fig_pie = px.pie(
                genre_counts, 
                values='Count', 
                names='Genre', 
                color_discrete_sequence=px.colors.sequential.RdBu,
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                legend=dict(font=dict(color='#ffffff')),
                margin=dict(t=10, b=10, l=10, r=10),
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c_col2:
            st.markdown("#### Audio Feature Signature")
            categories = ['Energy', 'Valence', 'Danceability', 'Acousticness']
            avg_vals = [
                playlist_df['energy'].mean(),
                playlist_df['valence'].mean(),
                playlist_df['danceability'].mean(),
                playlist_df['acousticness'].mean()
            ]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_vals,
                theta=categories,
                fill='toself',
                name='Playlist Average',
                line_color='#ec4899',
                fillcolor='rgba(236, 72, 153, 0.2)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='#8b949e')),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='#ffffff')),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=20, l=40, r=40),
                height=300
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        # Display the playlist tracks table with removal option
        st.markdown("#### Playlist Tracks")
        
        # Display each track with a remove button
        for index, row in playlist_df.iterrows():
            p_col1, p_col2, p_col3, p_col4 = st.columns([4, 3, 2, 1])
            with p_col1:
                st.markdown(f"🎵 **{row['title']}**")
            with p_col2:
                st.markdown(f"👤 {row['artist']}")
            with p_col3:
                st.markdown(f"🏷️ {row['genre']} | {row['release_year']}")
            with p_col4:
                st.button("🗑️ Remove", key=f"rem_{row['title']}_{index}", on_click=remove_from_playlist, args=(row['title'],), use_container_width=True)
            st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
            
        # Download Playlist as CSV
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = playlist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Playlist as CSV",
            data=csv_data,
            file_name="melodicmatch_playlist.csv",
            mime="text/csv",
            use_container_width=True
        )
