import requests
import pandas as pd

# Hardcoded API Key 
API_KEY = "AIzaSyDSsltnSEuvllUFCOK4EWsorfZWW45Ba_E"

def get_videos_from_playlist(playlist_id: str) -> list:
    """
    Mengambil daftar Video ID dari Playlist YouTube.
    """
    video_ids = []
    next_page_token = None

    while True:
        url = f"https://www.googleapis.com/youtube/v3/playlistItems"
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
    Mengambil statistik engagement video (views, likes, comments).
    """
    stats_list = []

    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        url = f"https://www.googleapis.com/youtube/v3/videos"
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
            stats_list.append({
                "video_id": item["id"],
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0))
            })

    return pd.DataFrame(stats_list)
