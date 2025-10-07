import streamlit as st
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import subprocess
import os
from io import BytesIO
import tempfile
import pydub
import speech_recognition as sr
from deep_translator import GoogleTranslator
import whisper
import time

# Note: Run `pip install streamlit pandas beautifulsoup4 requests pydub speechrecognition deep-translator whisper` before running.
# Ensure FFmpeg is installed (e.g., via `brew install ffmpeg` on macOS, `apt install ffmpeg` on Ubuntu, or download from https://ffmpeg.org/).

st.title("Robust Streaming Video Player with Chinese-to-English Captions")

@st.cache_data
def load_db(uploaded_file):
    """Load the uploaded .db file into a temporary SQLite connection."""
    if uploaded_file is not None:
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.write(uploaded_file.getvalue())
        temp_db.close()
        
        conn = sqlite3.connect(temp_db.name)
        try:
            df = pd.read_sql_query("SELECT title, author, duration, views, date, page_url FROM videos", conn)
            return df, temp_db.name
        except Exception as e:
            st.error(f"Error loading DB: {e}. Ensure the table 'videos' exists with columns: title, author, duration, views, date, page_url.")
            return None, None
        finally:
            conn.close()
    return None, None

def extract_m3u8_url(page_url):
    """Extract the HLS (m3u8) stream URL from the hsex.icu video page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(page_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if script.string and 'm3u8' in script.string:
                match = re.search(r'(?:src|url|link)\s*[:=]\s*["\']([^"\']*\.m3u8[^"\']*)["\']', script.string, re.IGNORECASE)
                if match:
                    m3u8_url = match.group(1)
                    if not m3u8_url.startswith('http'):
                        base_url = '/'.join(page_url.split('/')[:-1]) + '/'
                        m3u8_url = base_url + m3u8_url.lstrip('/')
                    return m3u8_url
        
        match = re.search(r'["\']([^"\']*\.m3u8[^"\']*)["\']', response.text)
        if match:
            m3u8_url = match.group(1)
            if not m3u8_url.startswith('http'):
                base_url = '/'.join(page_url.split('/')[:-1]) + '/'
                m3u8_url = base_url + m3u8_url.lstrip('/')
            return m3u8_url
        
        st.warning("Could not extract m3u8 URL. The site may have changed or use obfuscation.")
        return None
    except Exception as e:
        st.error(f"Error extracting m3u8: {e}")
        return None

def extract_audio_segment(m3u8_url, duration=60, output_filename="temp_audio.mp3"):
    """Extract a segment of audio from the video stream using FFmpeg."""
    try:
        cmd = [
            'ffmpeg',
            '-i', m3u8_url,
            '-t', str(duration),  # Limit to first 60 seconds
            '-vn',  # No video
            '-acodec', 'mp3',
            output_filename,
            '-y'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.returncode == 0:
            return output_filename
        else:
            st.error(f"FFmpeg audio extraction error: {result.stderr}")
            return None
    except subprocess.CalledProcessError as e:
        st.error(f"Audio extraction failed: {e}")
        return None
    except FileNotFoundError:
        st.error("FFmpeg not found. Please install FFmpeg.")
        return None

def transcribe_and_translate(audio_file):
    """Transcribe audio using Whisper and translate from Chinese to English."""
    try:
        # Load Whisper model (small for speed, supports Chinese)
        model = whisper.load_model("small")
        result = model.transcribe(audio_file, language="zh")
        transcribed_text = result["text"]
        
        # Translate to English
        translator = GoogleTranslator(source='zh-CN', target='en')
        translated_text = translator.translate(transcribed_text)
        
        # Create simple caption format (approximate timing)
        captions = []
        words = transcribed_text.split()
        for i, word in enumerate(words):
            start_time = i * 2  # Rough estimate: 2 seconds per word
            captions.append({
                'start': start_time,
                'end': start_time + 2,
                'text': translator.translate(word)
            })
        return captions, transcribed_text, translated_text
    except Exception as e:
        st.error(f"Transcription/Translation error: {e}")
        return [], "", ""

def download_video(m3u8_url, output_filename="downloaded_video.mp4"):
    """Download the video stream using FFmpeg."""
    try:
        cmd = [
            'ffmpeg',
            '-i', m3u8_url,
            '-c', 'copy',
            '-bsf:a', 'aac_adtstoasc',
            output_filename,
            '-y'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.returncode == 0:
            return True
        else:
            st.error(f"FFmpeg download error: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        st.error(f"Download failed: {e}")
        return False
    except FileNotFoundError:
        st.error("FFmpeg not found. Please install FFmpeg.")
        return False

# Main app logic
uploaded_file = st.file_uploader("Upload your cloud.db file containing video metadata", type="db")

if uploaded_file is not None:
    df, temp_db_path = load_db(uploaded_file)
    if df is not None and not df.empty:
        st.success(f"Loaded {len(df)} videos from DB.")
        
        st.subheader("Available Videos")
        st.dataframe(df[['title', 'author', 'duration', 'views', 'date']])
        
        selected_title = st.selectbox("Select a video to play:", df['title'].tolist())
        selected_row = df[df['title'] == selected_title].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Playing: {selected_row['title']}")
            st.write(f"**Author:** {selected_row['author']}")
            st.write(f"**Duration:** {selected_row['duration']}")
            st.write(f"**Views:** {selected_row['views']}")
            st.write(f"**Date:** {selected_row['date']}")
        
        page_url = selected_row['page_url']
        m3u8_url = extract_m3u8_url(page_url)
        
        if m3u8_url:
            with col2:
                st.video(m3u8_url)
            
            # Transcription and translation
            with st.spinner("Extracting and processing audio for captions..."):
                audio_file = extract_audio_segment(m3u8_url)
                if audio_file:
                    captions, transcribed_text, translated_text = transcribe_and_translate(audio_file)
                    os.unlink(audio_file)  # Clean up
                    if captions:
                        st.subheader("Captions (Chinese to English)")
                        # Display captions in a simple table
                        caption_df = pd.DataFrame(captions)
                        st.dataframe(caption_df[['start', 'end', 'text']])
                        st.write("**Original (Chinese):**")
                        st.write(transcribed_text)
                        st.write("**Translated (English):**")
                        st.write(translated_text)
                    else:
                        st.warning("No captions generated. Audio may be silent or not in Chinese.")
            
            # Download button
            if st.button("Download Video (records stream to MP4)"):
                with st.spinner("Downloading video stream..."):
                    output_file = f"{selected_title}.mp4".replace('/', '_').replace('\\', '_')
                    if download_video(m3u8_url, output_file):
                        with open(output_file, 'rb') as f:
                            video_bytes = f.read()
                        st.download_button(
                            label="Download Recorded Video",
                            data=video_bytes,
                            file_name=output_file,
                            mime="video/mp4"
                        )
                        os.unlink(output_file)
        else:
            st.error("Failed to extract stream URL. Please check the page manually.")
    
    if 'temp_db_path' in locals() and temp_db_path:
        try:
            os.unlink(temp_db_path)
        except:
            pass
else:
    st.info("Please upload your cloud.db file to get started. Ensure it contains a 'videos' table with columns like 'title', 'page_url', etc.")
