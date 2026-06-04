import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Curated dataset of 100 well-known songs (K-Pop, Pop, Indie, R&B, Rock, Hip-hop, Jazz, Classical)
# Audio features are rated 0.0 to 1.0:
# - energy: Intensity and activity (high = fast, loud, energetic; low = calm, quiet)
# - valence: Positivity / happiness (high = joyful, upbeat; low = sad, melancholic, serious)
# - danceability: How danceable it is based on tempo, rhythm, beat (high = groove/dance; low = irregular/flowy)
# - acousticness: Presence of acoustic instruments vs electronic synth (high = acoustic/piano; low = synth/electric guitar)
# - tempo_bpm: Actual tempo in Beats Per Minute (will be normalized for similarity calculations)

songs_data = [
    # --- K-POP / DANCE / HIP-HOP ---
    {
        "title": "Dynamite", "artist": "BTS", "genre": "K-Pop", "release_year": 2020, "mood": "Happy",
        "energy": 0.81, "valence": 0.96, "danceability": 0.75, "acousticness": 0.01, "tempo_bpm": 114,
        "cover_query": "bts-dynamite",
        "unsplash_url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=400&q=80",
        "youtube_query": "BTS Dynamite MV"
    },
    {
        "title": "Butter", "artist": "BTS", "genre": "K-Pop", "release_year": 2021, "mood": "Happy",
        "energy": 0.79, "valence": 0.91, "danceability": 0.82, "acousticness": 0.02, "tempo_bpm": 110,
        "cover_query": "butter",
        "unsplash_url": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=400&q=80",
        "youtube_query": "BTS Butter MV"
    },
    {
        "title": "Hype Boy", "artist": "NewJeans", "genre": "K-Pop", "release_year": 2022, "mood": "Chill",
        "energy": 0.72, "valence": 0.78, "danceability": 0.88, "acousticness": 0.15, "tempo_bpm": 100,
        "cover_query": "newjeans-hypeboy",
        "unsplash_url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=400&q=80",
        "youtube_query": "NewJeans Hype Boy MV"
    },
    {
        "title": "Ditto", "artist": "NewJeans", "genre": "K-Pop", "release_year": 2022, "mood": "Chill",
        "energy": 0.64, "valence": 0.55, "danceability": 0.81, "acousticness": 0.28, "tempo_bpm": 134,
        "cover_query": "retro-winter",
        "unsplash_url": "https://images.unsplash.com/photo-1487180142328-054b783fc471?w=400&q=80",
        "youtube_query": "NewJeans Ditto MV"
    },
    {
        "title": "Lilac", "artist": "IU", "genre": "K-Pop", "release_year": 2021, "mood": "Happy",
        "energy": 0.75, "valence": 0.82, "danceability": 0.74, "acousticness": 0.12, "tempo_bpm": 115,
        "cover_query": "lilac-flower",
        "unsplash_url": "https://images.unsplash.com/photo-1528605248644-14dd04022da1?w=400&q=80",
        "youtube_query": "IU Lilac MV"
    },
    {
        "title": "Love wins all", "artist": "IU", "genre": "K-Pop", "release_year": 2024, "mood": "Sad",
        "energy": 0.45, "valence": 0.35, "danceability": 0.42, "acousticness": 0.65, "tempo_bpm": 80,
        "cover_query": "love-wins",
        "unsplash_url": "https://images.unsplash.com/photo-1494905998402-395d579af36f?w=400&q=80",
        "youtube_query": "IU Love wins all MV"
    },
    {
        "title": "Through the Night", "artist": "IU", "genre": "K-Pop", "release_year": 2017, "mood": "Chill",
        "energy": 0.25, "valence": 0.40, "danceability": 0.58, "acousticness": 0.85, "tempo_bpm": 78,
        "cover_query": "starry-night",
        "unsplash_url": "https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?w=400&q=80",
        "youtube_query": "IU Through the Night MV"
    },
    {
        "title": "How You Like That", "artist": "BLACKPINK", "genre": "K-Pop", "release_year": 2020, "mood": "Energetic",
        "energy": 0.88, "valence": 0.64, "danceability": 0.83, "acousticness": 0.05, "tempo_bpm": 130,
        "cover_query": "neon-lights",
        "unsplash_url": "https://images.unsplash.com/photo-1516450360452-9312f5e86fc7?w=400&q=80",
        "youtube_query": "BLACKPINK How You Like That MV"
    },
    {
        "title": "Lovesick Girls", "artist": "BLACKPINK", "genre": "K-Pop", "release_year": 2020, "mood": "Energetic",
        "energy": 0.82, "valence": 0.72, "danceability": 0.78, "acousticness": 0.01, "tempo_bpm": 128,
        "cover_query": "cyberpunk-girl",
        "unsplash_url": "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=400&q=80",
        "youtube_query": "BLACKPINK Lovesick Girls MV"
    },
    {
        "title": "Love Dive", "artist": "IVE", "genre": "K-Pop", "release_year": 2022, "mood": "Energetic",
        "energy": 0.78, "valence": 0.68, "danceability": 0.80, "acousticness": 0.08, "tempo_bpm": 118,
        "cover_query": "pool-dive",
        "unsplash_url": "https://images.unsplash.com/photo-1519046904884-53103b34b206?w=400&q=80",
        "youtube_query": "IVE Love Dive MV"
    },
    {
        "title": "I AM", "artist": "IVE", "genre": "K-Pop", "release_year": 2023, "mood": "Happy",
        "energy": 0.88, "valence": 0.82, "danceability": 0.75, "acousticness": 0.02, "tempo_bpm": 126,
        "cover_query": "sky-high",
        "unsplash_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400&q=80",
        "youtube_query": "IVE I AM MV"
    },
    {
        "title": "Perfect Night", "artist": "LE SSERAFIM", "genre": "K-Pop", "release_year": 2023, "mood": "Chill",
        "energy": 0.68, "valence": 0.70, "danceability": 0.84, "acousticness": 0.06, "tempo_bpm": 115,
        "cover_query": "night-drive",
        "unsplash_url": "https://images.unsplash.com/photo-1492691527719-9d1e07e534b4?w=400&q=80",
        "youtube_query": "LE SSERAFIM Perfect Night MV"
    },

    # --- GLOBAL POP ---
    {
        "title": "Shake It Off", "artist": "Taylor Swift", "genre": "Pop", "release_year": 2014, "mood": "Happy",
        "energy": 0.80, "valence": 0.94, "danceability": 0.79, "acousticness": 0.06, "tempo_bpm": 160,
        "cover_query": "dancing-party",
        "unsplash_url": "https://images.unsplash.com/photo-1504609773096-104ff2c73ba4?w=400&q=80",
        "youtube_query": "Taylor Swift Shake It Off"
    },
    {
        "title": "Blank Space", "artist": "Taylor Swift", "genre": "Pop", "release_year": 2014, "mood": "Happy",
        "energy": 0.70, "valence": 0.75, "danceability": 0.76, "acousticness": 0.10, "tempo_bpm": 96,
        "cover_query": "fancy-room",
        "unsplash_url": "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=400&q=80",
        "youtube_query": "Taylor Swift Blank Space"
    },
    {
        "title": "Cardigan", "artist": "Taylor Swift", "genre": "Indie", "release_year": 2020, "mood": "Chill",
        "energy": 0.40, "valence": 0.50, "danceability": 0.61, "acousticness": 0.75, "tempo_bpm": 130,
        "cover_query": "cozy-sweater",
        "unsplash_url": "https://images.unsplash.com/photo-1583743814966-8936f5b7be1a?w=400&q=80",
        "youtube_query": "Taylor Swift cardigan"
    },
    {
        "title": "Bad Guy", "artist": "Billie Eilish", "genre": "Pop", "release_year": 2019, "mood": "Energetic",
        "energy": 0.43, "valence": 0.56, "danceability": 0.90, "acousticness": 0.33, "tempo_bpm": 135,
        "cover_query": "grunge-dark",
        "unsplash_url": "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=400&q=80",
        "youtube_query": "Billie Eilish bad guy"
    },
    {
        "title": "Ocean Eyes", "artist": "Billie Eilish", "genre": "Pop", "release_year": 2016, "mood": "Sad",
        "energy": 0.28, "valence": 0.17, "danceability": 0.60, "acousticness": 0.81, "tempo_bpm": 145,
        "cover_query": "ocean-mist",
        "unsplash_url": "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=400&q=80",
        "youtube_query": "Billie Eilish ocean eyes"
    },
    {
        "title": "What Was I Made For?", "artist": "Billie Eilish", "genre": "Pop", "release_year": 2023, "mood": "Sad",
        "energy": 0.10, "valence": 0.14, "danceability": 0.44, "acousticness": 0.96, "tempo_bpm": 78,
        "cover_query": "dollhouse",
        "unsplash_url": "https://images.unsplash.com/photo-1490730141103-6cac27aaab94?w=400&q=80",
        "youtube_query": "Billie Eilish What Was I Made For"
    },
    {
        "title": "Uptown Funk", "artist": "Bruno Mars", "genre": "Pop", "release_year": 2014, "mood": "Happy",
        "energy": 0.87, "valence": 0.93, "danceability": 0.85, "acousticness": 0.01, "tempo_bpm": 115,
        "cover_query": "retro-streets",
        "unsplash_url": "https://images.unsplash.com/photo-1511192336575-5a79af67a629?w=400&q=80",
        "youtube_query": "Mark Ronson Uptown Funk ft Bruno Mars"
    },
    {
        "title": "Just the Way You Are", "artist": "Bruno Mars", "genre": "Pop", "release_year": 2010, "mood": "Romantic",
        "energy": 0.84, "valence": 0.85, "danceability": 0.64, "acousticness": 0.02, "tempo_bpm": 109,
        "cover_query": "heart-light",
        "unsplash_url": "https://images.unsplash.com/photo-1518199266791-5375a83190b7?w=400&q=80",
        "youtube_query": "Bruno Mars Just The Way You Are"
    },
    {
        "title": "Leave the Door Open", "artist": "Silk Sonic", "genre": "R&B", "release_year": 2021, "mood": "Romantic",
        "energy": 0.62, "valence": 0.72, "danceability": 0.74, "acousticness": 0.18, "tempo_bpm": 82,
        "cover_query": "vintage-lounge",
        "unsplash_url": "https://images.unsplash.com/photo-1484755560693-a4074577af3a?w=400&q=80",
        "youtube_query": "Bruno Mars Anderson Paak Silk Sonic Leave The Door Open"
    },
    {
        "title": "Shape of You", "artist": "Ed Sheeran", "genre": "Pop", "release_year": 2017, "mood": "Happy",
        "energy": 0.65, "valence": 0.93, "danceability": 0.83, "acousticness": 0.58, "tempo_bpm": 96,
        "cover_query": "shape-geometric",
        "unsplash_url": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400&q=80",
        "youtube_query": "Ed Sheeran Shape of You"
    },
    {
        "title": "Perfect", "artist": "Ed Sheeran", "genre": "Pop", "release_year": 2017, "mood": "Romantic",
        "energy": 0.45, "valence": 0.59, "danceability": 0.60, "acousticness": 0.16, "tempo_bpm": 95,
        "cover_query": "couple-dancing",
        "unsplash_url": "https://images.unsplash.com/photo-1515934751635-c81c6bc9a2d8?w=400&q=80",
        "youtube_query": "Ed Sheeran Perfect"
    },
    {
        "title": "Blinding Lights", "artist": "The Weeknd", "genre": "Pop", "release_year": 2020, "mood": "Energetic",
        "energy": 0.80, "valence": 0.65, "danceability": 0.51, "acousticness": 0.00, "tempo_bpm": 171,
        "cover_query": "red-neon-car",
        "unsplash_url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=400&q=80",
        "youtube_query": "The Weeknd Blinding Lights"
    },
    {
        "title": "Starboy", "artist": "The Weeknd", "genre": "R&B", "release_year": 2016, "mood": "Chill",
        "energy": 0.67, "valence": 0.49, "danceability": 0.68, "acousticness": 0.14, "tempo_bpm": 186,
        "cover_query": "dark-luxury",
        "unsplash_url": "https://images.unsplash.com/photo-1533174072545-7a4b6ad7a6c3?w=400&q=80",
        "youtube_query": "The Weeknd Starboy ft Daft Punk"
    },
    {
        "title": "Levitating", "artist": "Dua Lipa", "genre": "Pop", "release_year": 2020, "mood": "Happy",
        "energy": 0.83, "valence": 0.91, "danceability": 0.70, "acousticness": 0.01, "tempo_bpm": 103,
        "cover_query": "disco-ball",
        "unsplash_url": "https://images.unsplash.com/photo-1482440308425-276ad0f28b19?w=400&q=80",
        "youtube_query": "Dua Lipa Levitating"
    },
    {
        "title": "Someone Like You", "artist": "Adele", "genre": "Pop", "release_year": 2011, "mood": "Sad",
        "energy": 0.33, "valence": 0.28, "danceability": 0.48, "acousticness": 0.89, "tempo_bpm": 135,
        "cover_query": "rainy-day-city",
        "unsplash_url": "https://images.unsplash.com/photo-1437419764061-2473afe69fc2?w=400&q=80",
        "youtube_query": "Adele Someone Like You"
    },
    {
        "title": "Rolling in the Deep", "artist": "Adele", "genre": "Pop", "release_year": 2011, "mood": "Energetic",
        "energy": 0.76, "valence": 0.52, "danceability": 0.73, "acousticness": 0.13, "tempo_bpm": 105,
        "cover_query": "fire-flame",
        "unsplash_url": "https://images.unsplash.com/photo-1549490349-8643362247b5?w=400&q=80",
        "youtube_query": "Adele Rolling in the Deep"
    },
    {
        "title": "Drivers License", "artist": "Olivia Rodrigo", "genre": "Pop", "release_year": 2021, "mood": "Sad",
        "energy": 0.43, "valence": 0.19, "danceability": 0.56, "acousticness": 0.77, "tempo_bpm": 144,
        "cover_query": "streetlights-driving",
        "unsplash_url": "https://images.unsplash.com/photo-1518609878373-06d740f60d8b?w=400&q=80",
        "youtube_query": "Olivia Rodrigo drivers license"
    },
    {
        "title": "Good 4 U", "artist": "Olivia Rodrigo", "genre": "Pop", "release_year": 2021, "mood": "Energetic",
        "energy": 0.89, "valence": 0.66, "danceability": 0.56, "acousticness": 0.02, "tempo_bpm": 167,
        "cover_query": "cheerleader",
        "unsplash_url": "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=400&q=80",
        "youtube_query": "Olivia Rodrigo good 4 u"
    },
    {
        "title": "As It Was", "artist": "Harry Styles", "genre": "Pop", "release_year": 2022, "mood": "Happy",
        "energy": 0.73, "valence": 0.66, "danceability": 0.52, "acousticness": 0.34, "tempo_bpm": 174,
        "cover_query": "colorful-hallway",
        "unsplash_url": "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400&q=80",
        "youtube_query": "Harry Styles As It Was"
    },
    {
        "title": "Someone You Loved", "artist": "Lewis Capaldi", "genre": "Pop", "release_year": 2018, "mood": "Sad",
        "energy": 0.41, "valence": 0.45, "danceability": 0.50, "acousticness": 0.75, "tempo_bpm": 110,
        "cover_query": "lonely-bench",
        "unsplash_url": "https://images.unsplash.com/photo-1475924156734-496f6cac6ec1?w=400&q=80",
        "youtube_query": "Lewis Capaldi Someone You Loved"
    },
    {
        "title": "Billie Jean", "artist": "Michael Jackson", "genre": "Pop", "release_year": 1982, "mood": "Energetic",
        "energy": 0.65, "valence": 0.86, "danceability": 0.92, "acousticness": 0.02, "tempo_bpm": 117,
        "cover_query": "fedora-hat",
        "unsplash_url": "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=400&q=80",
        "youtube_query": "Michael Jackson Billie Jean"
    },
    {
        "title": "Beat It", "artist": "Michael Jackson", "genre": "Pop", "release_year": 1982, "mood": "Energetic",
        "energy": 0.88, "valence": 0.90, "danceability": 0.81, "acousticness": 0.01, "tempo_bpm": 139,
        "cover_query": "leather-jacket",
        "unsplash_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&q=80",
        "youtube_query": "Michael Jackson Beat It"
    },
    {
        "title": "Get Lucky", "artist": "Daft Punk", "genre": "Pop", "release_year": 2013, "mood": "Happy",
        "energy": 0.81, "valence": 0.86, "danceability": 0.81, "acousticness": 0.04, "tempo_bpm": 116,
        "cover_query": "space-helmet",
        "unsplash_url": "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=400&q=80",
        "youtube_query": "Daft Punk Get Lucky"
    },
    {
        "title": "Happy", "artist": "Pharrell Williams", "genre": "Pop", "release_year": 2013, "mood": "Happy",
        "energy": 0.82, "valence": 0.96, "danceability": 0.65, "acousticness": 0.22, "tempo_bpm": 160,
        "cover_query": "happy-jumping",
        "unsplash_url": "https://images.unsplash.com/photo-1490730141103-6cac27aaab94?w=400&q=80",
        "youtube_query": "Pharrell Williams Happy"
    },
    {
        "title": "All of Me", "artist": "John Legend", "genre": "R&B", "release_year": 2013, "mood": "Romantic",
        "energy": 0.26, "valence": 0.33, "danceability": 0.42, "acousticness": 0.92, "tempo_bpm": 120,
        "cover_query": "grand-piano",
        "unsplash_url": "https://images.unsplash.com/photo-1520523839897-bd0b52f945a0?w=400&q=80",
        "youtube_query": "John Legend All of Me"
    },
    {
        "title": "Stay With Me", "artist": "Sam Smith", "genre": "Pop", "release_year": 2014, "mood": "Sad",
        "energy": 0.31, "valence": 0.22, "danceability": 0.42, "acousticness": 0.59, "tempo_bpm": 84,
        "cover_query": "gospel-choir",
        "unsplash_url": "https://images.unsplash.com/photo-1510915228340-29c85a43dcfe?w=400&q=80",
        "youtube_query": "Sam Smith Stay With Me"
    },
    {
        "title": "Peaches", "artist": "Justin Bieber", "genre": "Pop", "release_year": 2021, "mood": "Chill",
        "energy": 0.70, "valence": 0.73, "danceability": 0.68, "acousticness": 0.32, "tempo_bpm": 90,
        "cover_query": "peaches-fruit",
        "unsplash_url": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=400&q=80",
        "youtube_query": "Justin Bieber Peaches"
    },
    {
        "title": "Sugar", "artist": "Maroon 5", "genre": "Pop", "release_year": 2014, "mood": "Happy",
        "energy": 0.79, "valence": 0.88, "danceability": 0.75, "acousticness": 0.06, "tempo_bpm": 120,
        "cover_query": "wedding-cake",
        "unsplash_url": "https://images.unsplash.com/photo-1513151233558-d860c5398176?w=400&q=80",
        "youtube_query": "Maroon 5 Sugar"
    },
    {
        "title": "Memories", "artist": "Maroon 5", "genre": "Pop", "release_year": 2019, "mood": "Chill",
        "energy": 0.32, "valence": 0.57, "danceability": 0.76, "acousticness": 0.84, "tempo_bpm": 91,
        "cover_query": "polaroid-photos",
        "unsplash_url": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400&q=80",
        "youtube_query": "Maroon 5 Memories"
    },
    {
        "title": "Bad Romance", "artist": "Lady Gaga", "genre": "Pop", "release_year": 2009, "mood": "Energetic",
        "energy": 0.92, "valence": 0.71, "danceability": 0.70, "acousticness": 0.00, "tempo_bpm": 119,
        "cover_query": "glam-fashion",
        "unsplash_url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=400&q=80",
        "youtube_query": "Lady Gaga Bad Romance"
    },

    # --- ROCK / INDIE / ALTERNATIVE ---
    {
        "title": "Viva La Vida", "artist": "Coldplay", "genre": "Rock", "release_year": 2008, "mood": "Happy",
        "energy": 0.62, "valence": 0.42, "danceability": 0.49, "acousticness": 0.10, "tempo_bpm": 138,
        "cover_query": "royal-castle",
        "unsplash_url": "https://images.unsplash.com/photo-1507838153414-b4b713384a76?w=400&q=80",
        "youtube_query": "Coldplay Viva La Vida"
    },
    {
        "title": "Yellow", "artist": "Coldplay", "genre": "Rock", "release_year": 2000, "mood": "Chill",
        "energy": 0.46, "valence": 0.28, "danceability": 0.43, "acousticness": 0.00, "tempo_bpm": 173,
        "cover_query": "stars-yellow",
        "unsplash_url": "https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?w=400&q=80",
        "youtube_query": "Coldplay Yellow"
    },
    {
        "title": "Fix You", "artist": "Coldplay", "genre": "Rock", "release_year": 2005, "mood": "Sad",
        "energy": 0.42, "valence": 0.12, "danceability": 0.21, "acousticness": 0.16, "tempo_bpm": 138,
        "cover_query": "glowing-lights",
        "unsplash_url": "https://images.unsplash.com/photo-1492691527719-9d1e07e534b4?w=400&q=80",
        "youtube_query": "Coldplay Fix You"
    },
    {
        "title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "release_year": 1975, "mood": "Energetic",
        "energy": 0.40, "valence": 0.22, "danceability": 0.39, "acousticness": 0.27, "tempo_bpm": 144,
        "cover_query": "vintage-mic",
        "unsplash_url": "https://images.unsplash.com/photo-1465847899084-d164df4dedc6?w=400&q=80",
        "youtube_query": "Queen Bohemian Rhapsody"
    },
    {
        "title": "Don't Stop Me Now", "artist": "Queen", "genre": "Rock", "release_year": 1978, "mood": "Happy",
        "energy": 0.87, "valence": 0.71, "danceability": 0.56, "acousticness": 0.05, "tempo_bpm": 156,
        "cover_query": "shooting-star",
        "unsplash_url": "https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?w=400&q=80",
        "youtube_query": "Queen Don't Stop Me Now"
    },
    {
        "title": "Wonderwall", "artist": "Oasis", "genre": "Rock", "release_year": 1995, "mood": "Chill",
        "energy": 0.89, "valence": 0.43, "danceability": 0.38, "acousticness": 0.00, "tempo_bpm": 174,
        "cover_query": "acoustic-guitar-wall",
        "unsplash_url": "https://images.unsplash.com/photo-1510915361894-db8b60106cb1?w=400&q=80",
        "youtube_query": "Oasis Wonderwall"
    },
    {
        "title": "Back In Black", "artist": "AC/DC", "genre": "Rock", "release_year": 1880, "mood": "Energetic",
        "energy": 0.95, "valence": 0.76, "danceability": 0.69, "acousticness": 0.01, "tempo_bpm": 188,
        "cover_query": "black-leather",
        "unsplash_url": "https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=400&q=80",
        "youtube_query": "AC/DC Back In Black"
    },
    {
        "title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "Rock", "release_year": 1991, "mood": "Energetic",
        "energy": 0.91, "valence": 0.52, "danceability": 0.50, "acousticness": 0.00, "tempo_bpm": 117,
        "cover_query": "grunge-crowd",
        "unsplash_url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=400&q=80",
        "youtube_query": "Nirvana Smells Like Teen Spirit"
    },
    {
        "title": "Creep", "artist": "Radiohead", "genre": "Rock", "release_year": 1992, "mood": "Sad",
        "energy": 0.35, "valence": 0.10, "danceability": 0.51, "acousticness": 0.41, "tempo_bpm": 92,
        "cover_query": "rainy-neon",
        "unsplash_url": "https://images.unsplash.com/photo-1494253109108-2e30c049369b?w=400&q=80",
        "youtube_query": "Radiohead Creep"
    },
    {
        "title": "In the End", "artist": "Linkin Park", "genre": "Rock", "release_year": 2000, "mood": "Energetic",
        "energy": 0.86, "valence": 0.40, "danceability": 0.55, "acousticness": 0.01, "tempo_bpm": 105,
        "cover_query": "concrete-wall",
        "unsplash_url": "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=400&q=80",
        "youtube_query": "Linkin Park In The End"
    },
    {
        "title": "Sweet Child O' Mine", "artist": "Guns N' Roses", "genre": "Rock", "release_year": 1987, "mood": "Energetic",
        "energy": 0.90, "valence": 0.62, "danceability": 0.45, "acousticness": 0.09, "tempo_bpm": 128,
        "cover_query": "rose-tattoo",
        "unsplash_url": "https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=400&q=80",
        "youtube_query": "Guns N Roses Sweet Child O Mine"
    },
    {
        "title": "Wherever you are", "artist": "ONE OK ROCK", "genre": "Rock", "release_year": 2010, "mood": "Romantic",
        "energy": 0.68, "valence": 0.45, "danceability": 0.52, "acousticness": 0.11, "tempo_bpm": 85,
        "cover_query": "stage-spotlight",
        "unsplash_url": "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=400&q=80",
        "youtube_query": "ONE OK ROCK Wherever you are"
    },

    # --- JAZZ ---
    {
        "title": "What A Wonderful World", "artist": "Louis Armstrong", "genre": "Jazz", "release_year": 1967, "mood": "Chill",
        "energy": 0.15, "valence": 0.49, "danceability": 0.38, "acousticness": 0.95, "tempo_bpm": 77,
        "cover_query": "green-world",
        "unsplash_url": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=400&q=80",
        "youtube_query": "Louis Armstrong What A Wonderful World"
    },
    {
        "title": "So What", "artist": "Miles Davis", "genre": "Jazz", "release_year": 1959, "mood": "Focus",
        "energy": 0.20, "valence": 0.34, "danceability": 0.55, "acousticness": 0.85, "tempo_bpm": 133,
        "cover_query": "trumpet",
        "unsplash_url": "https://images.unsplash.com/photo-1511192336575-5a79af67a629?w=400&q=80",
        "youtube_query": "Miles Davis So What"
    },
    {
        "title": "Take Five", "artist": "Dave Brubeck", "genre": "Jazz", "release_year": 1959, "mood": "Focus",
        "energy": 0.28, "valence": 0.55, "danceability": 0.64, "acousticness": 0.88, "tempo_bpm": 174,
        "cover_query": "saxophone",
        "unsplash_url": "https://images.unsplash.com/photo-1525994886773-080587e161c2?w=400&q=80",
        "youtube_query": "Dave Brubeck Take Five"
    },
    {
        "title": "Blue in Green", "artist": "Bill Evans", "genre": "Jazz", "release_year": 1959, "mood": "Sad",
        "energy": 0.08, "valence": 0.15, "danceability": 0.40, "acousticness": 0.97, "tempo_bpm": 65,
        "cover_query": "dim-jazz-bar",
        "unsplash_url": "https://images.unsplash.com/photo-1481137314488-8543362247b5?w=400&q=80",
        "youtube_query": "Bill Evans Blue in Green"
    },
    {
        "title": "Don't Know Why", "artist": "Norah Jones", "genre": "Jazz", "release_year": 2002, "mood": "Chill",
        "energy": 0.18, "valence": 0.58, "danceability": 0.73, "acousticness": 0.88, "tempo_bpm": 88,
        "cover_query": "rainy-coffee",
        "unsplash_url": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=400&q=80",
        "youtube_query": "Norah Jones Don't Know Why"
    },
    {
        "title": "Fly Me to the Moon", "artist": "Frank Sinatra", "genre": "Jazz", "release_year": 1964, "mood": "Romantic",
        "energy": 0.32, "valence": 0.65, "danceability": 0.62, "acousticness": 0.70, "tempo_bpm": 119,
        "cover_query": "moon-classic",
        "unsplash_url": "https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?w=400&q=80",
        "youtube_query": "Frank Sinatra Fly Me to the Moon"
    },
    {
        "title": "Dream a Little Dream of Me", "artist": "Ella Fitzgerald", "genre": "Jazz", "release_year": 1950, "mood": "Romantic",
        "energy": 0.12, "valence": 0.44, "danceability": 0.48, "acousticness": 0.92, "tempo_bpm": 78,
        "cover_query": "cloudy-dream",
        "unsplash_url": "https://images.unsplash.com/photo-1490730141103-6cac27aaab94?w=400&q=80",
        "youtube_query": "Ella Fitzgerald Dream a Little Dream of Me"
    },
    {
        "title": "I Fall in Love Too Easily", "artist": "Chet Baker", "genre": "Jazz", "release_year": 1953, "mood": "Sad",
        "energy": 0.05, "valence": 0.22, "danceability": 0.35, "acousticness": 0.98, "tempo_bpm": 70,
        "cover_query": "trumpet-crying",
        "unsplash_url": "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=400&q=80",
        "youtube_query": "Chet Baker I Fall in Love Too Easily"
    },

    # --- CLASSICAL ---
    {
        "title": "Moonlight Sonata", "artist": "Ludwig van Beethoven", "genre": "Classical", "release_year": 1801, "mood": "Sad",
        "energy": 0.02, "valence": 0.10, "danceability": 0.30, "acousticness": 0.99, "tempo_bpm": 60,
        "cover_query": "moon-water",
        "unsplash_url": "https://images.unsplash.com/photo-1508247967583-7d982ea00926?w=400&q=80",
        "youtube_query": "Beethoven Moonlight Sonata 1st movement"
    },
    {
        "title": "Eine kleine Nachtmusik", "artist": "Wolfgang Amadeus Mozart", "genre": "Classical", "release_year": 1787, "mood": "Happy",
        "energy": 0.55, "valence": 0.81, "danceability": 0.52, "acousticness": 0.94, "tempo_bpm": 135,
        "cover_query": "gold-palace",
        "unsplash_url": "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400&q=80",
        "youtube_query": "Mozart Eine kleine Nachtmusik"
    },
    {
        "title": "Cello Suite No. 1", "artist": "Johann Sebastian Bach", "genre": "Classical", "release_year": 1720, "mood": "Focus",
        "energy": 0.15, "valence": 0.42, "danceability": 0.40, "acousticness": 0.98, "tempo_bpm": 80,
        "cover_query": "wooden-cello",
        "unsplash_url": "https://images.unsplash.com/photo-1507838153414-b4b713384a76?w=400&q=80",
        "youtube_query": "Bach Cello Suite No 1 Yo-Yo Ma"
    },
    {
        "title": "Nocturne Op. 9 No. 2", "artist": "Frédéric Chopin", "genre": "Classical", "release_year": 1832, "mood": "Romantic",
        "energy": 0.04, "valence": 0.25, "danceability": 0.35, "acousticness": 0.99, "tempo_bpm": 65,
        "cover_query": "candelabra-piano",
        "unsplash_url": "https://images.unsplash.com/photo-1520523839897-bd0b52f945a0?w=400&q=80",
        "youtube_query": "Chopin Nocturne Op 9 No 2 Rubinstein"
    },
    {
        "title": "Clair de Lune", "artist": "Claude Debussy", "genre": "Classical", "release_year": 1905, "mood": "Romantic",
        "energy": 0.03, "valence": 0.20, "danceability": 0.32, "acousticness": 0.99, "tempo_bpm": 62,
        "cover_query": "moonlight-sea",
        "unsplash_url": "https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?w=400&q=80",
        "youtube_query": "Debussy Clair de Lune piano"
    },
    {
        "title": "Spring (The Four Seasons)", "artist": "Antonio Vivaldi", "genre": "Classical", "release_year": 1725, "mood": "Happy",
        "energy": 0.65, "valence": 0.85, "danceability": 0.45, "acousticness": 0.90, "tempo_bpm": 105,
        "cover_query": "flower-garden",
        "unsplash_url": "https://images.unsplash.com/photo-1528605248644-14dd04022da1?w=400&q=80",
        "youtube_query": "Vivaldi Four Seasons Spring"
    },

    # --- K-INDIE / K-BALLAD / R&B ---
    {
        "title": "Love Lee", "artist": "AKMU", "genre": "K-Pop", "release_year": 2023, "mood": "Happy",
        "energy": 0.68, "valence": 0.89, "danceability": 0.78, "acousticness": 0.45, "tempo_bpm": 112,
        "cover_query": "heart-pink",
        "unsplash_url": "https://images.unsplash.com/photo-1518199266791-5375a83190b7?w=400&q=80",
        "youtube_query": "AKMU Love Lee MV"
    },
    {
        "title": "How can I love the heartbreak, you`re the one I love", "artist": "AKMU", "genre": "Indie", "release_year": 2019, "mood": "Sad",
        "energy": 0.28, "valence": 0.28, "danceability": 0.48, "acousticness": 0.82, "tempo_bpm": 80,
        "cover_query": "sea-sunset",
        "unsplash_url": "https://images.unsplash.com/photo-1490730141103-6cac27aaab94?w=400&q=80",
        "youtube_query": "AKMU How can I love the heartbreak MV"
    },
    {
        "title": "For Lovers Who Hesitate", "artist": "Jannabi", "genre": "Indie", "release_year": 2019, "mood": "Romantic",
        "energy": 0.45, "valence": 0.48, "danceability": 0.55, "acousticness": 0.68, "tempo_bpm": 105,
        "cover_query": "retro-record",
        "unsplash_url": "https://images.unsplash.com/photo-1484755560693-a4074577af3a?w=400&q=80",
        "youtube_query": "Jannabi For Lovers Who Hesitate MV"
    },
    {
        "title": "Graduation", "artist": "10cm", "genre": "Indie", "release_year": 2023, "mood": "Sad",
        "energy": 0.35, "valence": 0.45, "danceability": 0.62, "acousticness": 0.72, "tempo_bpm": 88,
        "cover_query": "school-yard",
        "unsplash_url": "https://images.unsplash.com/photo-1528605248644-14dd04022da1?w=400&q=80",
        "youtube_query": "10cm Graduation MV"
    },
    {
        "title": "To My Youth", "artist": "BOL4", "genre": "Indie", "release_year": 2017, "mood": "Sad",
        "energy": 0.42, "valence": 0.35, "danceability": 0.50, "acousticness": 0.78, "tempo_bpm": 92,
        "cover_query": "teens-sky",
        "unsplash_url": "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400&q=80",
        "youtube_query": "BOL4 To My Youth MV"
    },
    {
        "title": "Galaxy", "artist": "BOL4", "genre": "Indie", "release_year": 2016, "mood": "Romantic",
        "energy": 0.68, "valence": 0.78, "danceability": 0.70, "acousticness": 0.38, "tempo_bpm": 110,
        "cover_query": "starry-universe",
        "unsplash_url": "https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?w=400&q=80",
        "youtube_query": "BOL4 Galaxy MV"
    },
    {
        "title": "You, Clouds, Rain", "artist": "Heize", "genre": "R&B", "release_year": 2017, "mood": "Sad",
        "energy": 0.42, "valence": 0.32, "danceability": 0.68, "acousticness": 0.52, "tempo_bpm": 83,
        "cover_query": "umbrella-rain",
        "unsplash_url": "https://images.unsplash.com/photo-1437419764061-2473afe69fc2?w=400&q=80",
        "youtube_query": "Heize You Clouds Rain MV"
    },
    {
        "title": "Beautiful", "artist": "Crush", "genre": "R&B", "release_year": 2016, "mood": "Romantic",
        "energy": 0.38, "valence": 0.42, "danceability": 0.45, "acousticness": 0.75, "tempo_bpm": 74,
        "cover_query": "k-drama-snow",
        "unsplash_url": "https://images.unsplash.com/photo-1494905998402-395d579af36f?w=400&q=80",
        "youtube_query": "Crush Beautiful Goblin OST"
    },
    {
        "title": "instagram", "artist": "DEAN", "genre": "R&B", "release_year": 2017, "mood": "Sad",
        "energy": 0.46, "valence": 0.49, "danceability": 0.75, "acousticness": 0.55, "tempo_bpm": 105,
        "cover_query": "phone-screen",
        "unsplash_url": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=400&q=80",
        "youtube_query": "DEAN instagram MV"
    },
    {
        "title": "Any Song", "artist": "Zico", "genre": "K-Pop", "release_year": 2020, "mood": "Happy",
        "energy": 0.71, "valence": 0.88, "danceability": 0.88, "acousticness": 0.08, "tempo_bpm": 108,
        "cover_query": "party-group",
        "unsplash_url": "https://images.unsplash.com/photo-1504609773096-104ff2c73ba4?w=400&q=80",
        "youtube_query": "ZICO Any Song MV"
    },
    {
        "title": "Old Love", "artist": "Lee Moon-sae", "genre": "K-Pop", "release_year": 1991, "mood": "Sad",
        "energy": 0.21, "valence": 0.25, "danceability": 0.42, "acousticness": 0.88, "tempo_bpm": 75,
        "cover_query": "retro-alley",
        "unsplash_url": "https://images.unsplash.com/photo-1487180142328-054b783fc471?w=400&q=80",
        "youtube_query": "Lee Moon sae Old Love"
    },
    {
        "title": "Letter from a Private", "artist": "Kim Kwang-seok", "genre": "Indie", "release_year": 1993, "mood": "Sad",
        "energy": 0.18, "valence": 0.20, "danceability": 0.38, "acousticness": 0.90, "tempo_bpm": 80,
        "cover_query": "soldier-letter",
        "unsplash_url": "https://images.unsplash.com/photo-1475924156734-496f6cac6ec1?w=400&q=80",
        "youtube_query": "Kim Kwang seok Letter from a Private"
    },
    {
        "title": "On the Road", "artist": "Sung Si-kyung", "genre": "K-Pop", "release_year": 2006, "mood": "Romantic",
        "energy": 0.35, "valence": 0.33, "danceability": 0.44, "acousticness": 0.78, "tempo_bpm": 78,
        "cover_query": "fall-road",
        "unsplash_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400&q=80",
        "youtube_query": "Sung Si kyung On the Road"
    },
    {
        "title": "Me After You", "artist": "Paul Kim", "genre": "K-Pop", "release_year": 2018, "mood": "Romantic",
        "energy": 0.32, "valence": 0.38, "danceability": 0.46, "acousticness": 0.81, "tempo_bpm": 76,
        "cover_query": "couple-hands",
        "unsplash_url": "https://images.unsplash.com/photo-1518199266791-5375a83190b7?w=400&q=80",
        "youtube_query": "Paul Kim Me After You MV"
    },
    {
        "title": "Gift", "artist": "MeloMance", "genre": "K-Pop", "release_year": 2017, "mood": "Happy",
        "energy": 0.58, "valence": 0.75, "danceability": 0.65, "acousticness": 0.45, "tempo_bpm": 105,
        "cover_query": "present-box",
        "unsplash_url": "https://images.unsplash.com/photo-1513151233558-d860c5398176?w=400&q=80",
        "youtube_query": "MeloMance Gift MV"
    },
    {
        "title": "Trust in Me", "artist": "Lim Young-woong", "genre": "K-Pop", "release_year": 2020, "mood": "Romantic",
        "energy": 0.38, "valence": 0.31, "danceability": 0.48, "acousticness": 0.74, "tempo_bpm": 72,
        "cover_query": "warm-sunlight",
        "unsplash_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400&q=80",
        "youtube_query": "Lim Young woong Trust in Me"
    },
    {
        "title": "Crooked", "artist": "G-Dragon", "genre": "K-Pop", "release_year": 2013, "mood": "Energetic",
        "energy": 0.90, "valence": 0.72, "danceability": 0.72, "acousticness": 0.02, "tempo_bpm": 128,
        "cover_query": "neon-rebel",
        "unsplash_url": "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=400&q=80",
        "youtube_query": "G-DRAGON Crooked MV"
    },

    # --- HIP-HOP / GLOBAL RAP ---
    {
        "title": "Lose Yourself", "artist": "Eminem", "genre": "Hip-Hop", "release_year": 2002, "mood": "Energetic",
        "energy": 0.89, "valence": 0.59, "danceability": 0.69, "acousticness": 0.01, "tempo_bpm": 86,
        "cover_query": "hoodie-underground",
        "unsplash_url": "https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=400&q=80",
        "youtube_query": "Eminem Lose Yourself"
    },
    {
        "title": "HUMBLE.", "artist": "Kendrick Lamar", "genre": "Hip-Hop", "release_year": 2017, "mood": "Energetic",
        "energy": 0.81, "valence": 0.42, "danceability": 0.91, "acousticness": 0.00, "tempo_bpm": 150,
        "cover_query": "crown-graffiti",
        "unsplash_url": "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=400&q=80",
        "youtube_query": "Kendrick Lamar HUMBLE"
    },
    {
        "title": "Hotline Bling", "artist": "Drake", "genre": "Hip-Hop", "release_year": 2015, "mood": "Chill",
        "energy": 0.62, "valence": 0.55, "danceability": 0.90, "acousticness": 0.01, "tempo_bpm": 135,
        "cover_query": "pastel-background",
        "unsplash_url": "https://images.unsplash.com/photo-1482440308425-276ad0f28b19?w=400&q=80",
        "youtube_query": "Drake Hotline Bling MV"
    },
    {
        "title": "Sunflower", "artist": "Post Malone", "genre": "Hip-Hop", "release_year": 2018, "mood": "Happy",
        "energy": 0.48, "valence": 0.91, "danceability": 0.76, "acousticness": 0.56, "tempo_bpm": 90,
        "cover_query": "sunflower-field",
        "unsplash_url": "https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=400&q=80",
        "youtube_query": "Post Malone Swae Lee Sunflower"
    },
    {
        "title": "Circles", "artist": "Post Malone", "genre": "Pop", "release_year": 2019, "mood": "Chill",
        "energy": 0.76, "valence": 0.55, "danceability": 0.70, "acousticness": 0.19, "tempo_bpm": 120,
        "cover_query": "forest-fog",
        "unsplash_url": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=400&q=80",
        "youtube_query": "Post Malone Circles"
    },
    {
        "title": "ILYSB", "artist": "LANY", "genre": "Indie", "release_year": 2017, "mood": "Romantic",
        "energy": 0.48, "valence": 0.68, "danceability": 0.65, "acousticness": 0.12, "tempo_bpm": 102,
        "cover_query": "neon-sign-love",
        "unsplash_url": "https://images.unsplash.com/photo-1518199266791-5375a83190b7?w=400&q=80",
        "youtube_query": "LANY ILYSB"
    },
    {
        "title": "I Like Me Better", "artist": "Lauv", "genre": "Pop", "release_year": 2017, "mood": "Happy",
        "energy": 0.50, "valence": 0.75, "danceability": 0.75, "acousticness": 0.53, "tempo_bpm": 92,
        "cover_query": "nyc-street",
        "unsplash_url": "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=400&q=80",
        "youtube_query": "Lauv I Like Me Better"
    },
    {
        "title": "Self Care", "artist": "Mac Miller", "genre": "Hip-Hop", "release_year": 2018, "mood": "Chill",
        "energy": 0.52, "valence": 0.40, "danceability": 0.72, "acousticness": 0.10, "tempo_bpm": 120,
        "cover_query": "swimming-pool-lone",
        "unsplash_url": "https://images.unsplash.com/photo-1519046904884-53103b34b206?w=400&q=80",
        "youtube_query": "Mac Miller Self Care MV"
    },
    {
        "title": "Square (2017)", "artist": "Yerin Baek", "genre": "Indie", "release_year": 2019, "mood": "Happy",
        "energy": 0.68, "valence": 0.78, "danceability": 0.62, "acousticness": 0.35, "tempo_bpm": 115,
        "cover_query": "city-rooftop-sun",
        "unsplash_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400&q=80",
        "youtube_query": "Yerin Baek Square 2017"
    },
    {
        "title": "Scars Leave Beautiful Trace", "artist": "Car, the garden", "genre": "Indie", "release_year": 2019, "mood": "Sad",
        "energy": 0.32, "valence": 0.28, "danceability": 0.44, "acousticness": 0.75, "tempo_bpm": 80,
        "cover_query": "sadness-sunset",
        "unsplash_url": "https://images.unsplash.com/photo-1490730141103-6cac27aaab94?w=400&q=80",
        "youtube_query": "Car the garden Scars Leave Beautiful Trace"
    },
    {
        "title": "Glimpse of Us", "artist": "Joji", "genre": "Indie", "release_year": 2022, "mood": "Sad",
        "energy": 0.18, "valence": 0.12, "danceability": 0.44, "acousticness": 0.95, "tempo_bpm": 73,
        "cover_query": "rainy-car-window",
        "unsplash_url": "https://images.unsplash.com/photo-1437419764061-2473afe69fc2?w=400&q=80",
        "youtube_query": "Joji Glimpse of Us MV"
    },
    {
        "title": "Bam Yang Gang", "artist": "BIBI", "genre": "K-Pop", "release_year": 2024, "mood": "Happy",
        "energy": 0.52, "valence": 0.76, "danceability": 0.81, "acousticness": 0.55, "tempo_bpm": 122,
        "cover_query": "sweet-jelly",
        "unsplash_url": "https://images.unsplash.com/photo-1601004890684-d8cbf643f5f2?w=400&q=80",
        "youtube_query": "BIBI Bam Yang Gang MV"
    },
    {
        "title": "plot twist", "artist": "TWS", "genre": "K-Pop", "release_year": 2024, "mood": "Happy",
        "energy": 0.85, "valence": 0.90, "danceability": 0.78, "acousticness": 0.11, "tempo_bpm": 125,
        "cover_query": "school-friends",
        "unsplash_url": "https://images.unsplash.com/photo-1528605248644-14dd04022da1?w=400&q=80",
        "youtube_query": "TWS plot twist MV"
    },
    {
        "title": "Get A Guitar", "artist": "RIIZE", "genre": "K-Pop", "release_year": 2023, "mood": "Happy",
        "energy": 0.82, "valence": 0.88, "danceability": 0.84, "acousticness": 0.08, "tempo_bpm": 115,
        "cover_query": "skate-guitar",
        "unsplash_url": "https://images.unsplash.com/photo-1510915361894-db8b60106cb1?w=400&q=80",
        "youtube_query": "RIIZE Get A Guitar MV"
    }
]

# Load dataset into pandas DataFrame
df_songs = pd.DataFrame(songs_data)

# Normalize the tempo feature for distance calculations
scaler = MinMaxScaler()
df_songs['normalized_tempo'] = scaler.fit_transform(df_songs[['tempo_bpm']])

# Numeric features list for similarity calculations
FEATURE_COLS = ['energy', 'valence', 'danceability', 'acousticness', 'normalized_tempo']

def get_recommendations_by_song(song_title, top_n=5):
    """
    Finds the top N most similar songs to the selected song_title
    using Cosine Similarity on audio features.
    """
    if song_title not in df_songs['title'].values:
        return pd.DataFrame()

    # Find the target song features
    target_idx = df_songs[df_songs['title'] == song_title].index[0]
    
    # Calculate similarity matrix of target song vs all other songs
    features_matrix = df_songs[FEATURE_COLS].values
    target_features = features_matrix[target_idx].reshape(1, -1)
    
    sim_scores = cosine_similarity(target_features, features_matrix)[0]
    
    # Add similarity scores to a copy of the dataframe
    df_copy = df_songs.copy()
    df_copy['similarity'] = sim_scores
    
    # Sort by similarity and drop the target song itself
    recommendations = df_copy.sort_values(by='similarity', ascending=False)
    recommendations = recommendations[recommendations.index != target_idx]
    
    return recommendations.head(top_n)

def get_recommendations_by_filters(mood=None, genre=None, start_year=None, end_year=None, sort_by=None):
    """
    Filters songs by mood, genre, and release year range.
    Optionally sorts by a specific audio feature.
    """
    filtered_df = df_songs.copy()
    
    if mood and mood != "All":
        filtered_df = filtered_df[filtered_df['mood'] == mood]
        
    if genre and genre != "All":
        filtered_df = filtered_df[filtered_df['genre'] == genre]
        
    if start_year is not None:
        filtered_df = filtered_df[filtered_df['release_year'] >= start_year]
        
    if end_year is not None:
        filtered_df = filtered_df[filtered_df['release_year'] <= end_year]
        
    if sort_by and sort_by in FEATURE_COLS:
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
    elif sort_by == 'tempo_bpm':
        filtered_df = filtered_df.sort_values(by='tempo_bpm', ascending=False)
        
    return filtered_df
