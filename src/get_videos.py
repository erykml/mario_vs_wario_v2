# script used for downloading the source videos from YouTube

from pytube import YouTube
import os 
from config import VIDEOS_DIR, YT_LINKS_DIR, CLASSES


def download_videos_from_yt():

    os.makedirs(os.path.dirname(VIDEOS_DIR), exist_ok=True)

    for class_name in CLASSES:

        print(f"Downloading the {class_name} video...")
    
        yt = YouTube(YT_LINKS_DIR[class_name])

        (
            yt
            .streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
            .download(VIDEOS_DIR)
        )

        print("done.")

if __name__ == "__main__":
    download_videos_from_yt()