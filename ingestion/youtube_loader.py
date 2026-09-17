import yt_dlp
import whisper
import os


class YouTubeLoader:
    def __init__(self):
        self.model = whisper.load_model("base")

    def download_audio(self, url):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'audio.%(ext)s',
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return "audio.webm"

    def fetch_and_clean(self, url):
        try:
            audio_path = self.download_audio(url)

            result = self.model.transcribe(audio_path)
            text = result["text"]

            os.remove(audio_path)

            return text

        except Exception as e:
            print(f"Error: {str(e)}")
            return None