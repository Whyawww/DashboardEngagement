import requests
import pandas as pd
import numpy as np

API_KEY = "AIzaSyDSsltnSEuvllUFCOK4EWsorfZWW45Ba_E"

def get_videos_from_playlist(playlist_id: str) -> list:
    """
    Mengambil daftar Video ID dari Playlist YouTube.

    Args:
        playlist_id (str): ID dari playlist YouTube.

    Returns:
        list: Daftar video ID.
    """
    video_ids = []
    next_page_token = None

    while True:
        url = "https://www.googleapis.com/youtube/v3/playlistItems"
        params = {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": 50,
            "key": API_KEY
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(url, params=params).json()

        if "error" in response:
            raise Exception(f"Error fetching playlist: {response['error']['message']}")

        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids

def get_video_statistics(video_ids: list) -> pd.DataFrame:
    """
    Mengambil statistik engagement video (views, likes, comments),
    dan menghitung engagement_rate.

    Args:
        video_ids (list): Daftar video ID.

    Returns:
        pd.DataFrame: Dataframe berisi statistik dan engagement rate.
    """
    stats_list = []

    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "statistics",
            "id": ",".join(batch_ids),
            "key": API_KEY
        }

        response = requests.get(url, params=params).json()

        if "error" in response:
            raise Exception(f"Error fetching video statistics: {response['error']['message']}")

        for item in response.get("items", []):
            stats = item.get("statistics", {})
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))

            # Hitung engagement_rate dengan aman
            if views > 0:
                engagement_rate = (likes + comments) / views
            else:
                engagement_rate = 0

            stats_list.append({
                "video_id": item["id"],
                "views": views,
                "likes": likes,
                "comments": comments,
                "engagement_rate": engagement_rate
            })

    df_stats = pd.DataFrame(stats_list)

    # Tangani NaN atau inf jika ada
    df_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_stats.fillna(0, inplace=True)

    return df_stats
