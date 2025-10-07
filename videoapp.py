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

# Note: Run `pip install streamlit pandas beautifulsoup4 requests pydub speechrecognition deep-translator openai-whisper`
# Ensure FFmpeg is installed (e.g., `brew install ffmpeg` on macOS, `apt install ffmpeg` on Ubuntu, or download from https://ffmpeg.org/)

st.title("Robust Streaming Video Player with Chinese-to-English Captions")

def init_db():
    """Initialize an in-memory SQLite database to store video metadata."""
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS videos
                 (title TEXT, author TEXT, duration TEXT, views TEXT, date TEXT, page_url TEXT)''')
    conn.commit()
    return conn

def extract_metadata(page_url):
    """Scrape metadata from the hsex.icu video page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(page_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract metadata (adjust selectors based on site structure)
        title = soup.find('h1') or soup.find('title') or 'Unknown Title'
        title = title.text.strip() if title else 'Unknown Title'
        
        author = 'Unknown Author'
        author_elem = soup.find(string=re.compile('作者', re.I))
        if author_elem:
            author = author_elem.split('：')[-1].strip() if '：' in author_elem else author_elem.strip()
        
        duration = 'Unknown Duration'
        duration_elem = soup.find(string=re.compile(r'\d{2}:\d{2}', re.I))
        if duration_elem:
            duration = duration_elem.strip()
        
        views = 'Unknown Views'
        views_elem = soup.find(string=re.compile(r'\d+\.?\d*k?', re.I))
        if views_elem:
            views = views_elem.strip()
        
        date = 'Unknown Date'
        date_elem = soup.find(string=re.compile(r'\d+天前|日期', re.I))
        if date_elem:
            date = date_elem.strip()
        
        return {
            'title': title,
            'author': author,
            'duration': duration,
            'views': views,
            'date': date,
            'page_url': page_url
        }
    except Exception as e:
        st.error(f"Error scraping metadata: {e}")
        return None

def save_metadata_to_db(conn, metadata):
    """Save scraped metadata to the in-memory database."""
    c = conn.cursor()
    c.execute('''INSERT INTO videos (title, author, duration, views, date, page_url)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (metadata['title'], metadata['author'], metadata['duration'],
               metadata['views'], metadata['date'], metadata['page_url']))
    conn.commit()

def load_db(conn):
    """Load metadata from the in-memory database."""
    try:
        df = pd.read_sql_query("SELECT title, author, duration, views, date, page_url FROM videos", conn)
        return df
    except Exception as e:
        st.error(f"Error loading DB: {e}")
        return None

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
            '-t', str(duration),
            '-vn',
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
        model = whisper.load_model("small")
        result = model.transcribe(audio_file, language="zh")
        transcribed_text = result["text"]
        
        translator = GoogleTranslator(source='zh-CN', target='en')
        translated_text = translator.translate(transcribed_text)
        
        captions = []
        words = transcribed_text.split()
        for i, word in enumerate(words):
            start_time = i * 2
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

# Initialize session state for database connection
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_db()

# Main app logic
st.subheader("Enter Video URL")
video_url = st.text_input("Paste the hsex.icu video URL (e.g., https://hsex.icu/video-1131934.htm):")
if st.button("Load Video"):
    if video_url:
        with st.spinner("Scraping metadata..."):
            metadata = extract_metadata(video_url)
            if metadata:
                save_metadata_to_db(st.session_state.db_conn, metadata)
                df = load_db(st.session_state.db_conn)
                if df is not None and not df.empty:
                    st.success("Metadata loaded successfully!")
                    
                    st.subheader("Video Metadata")
                    st.dataframe(df[['title', 'author', 'duration', 'views', 'date']])
                    
                    selected_row = df.iloc[-1]  # Use the latest added video
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
                        
                        with st.spinner("Extracting and processing audio for captions..."):
                            audio_file = extract_audio_segment(m3u8_url)
                            if audio_file:
                                captions, transcribed_text, translated_text = transcribe_and_translate(audio_file)
                                os.unlink(audio_file)
                                if captions:
                                    st.subheader("Captions (Chinese to English)")
                                    caption_df = pd.DataFrame(captions)
                                    st.dataframe(caption_df[['start', 'end', 'text']])
                                    st.write("**Original (Chinese):**")
                                    st.write(transcribed_text)
                                    st.write("**Translated (English):**")
                                    st.write(translated_text)
                                else:
                                    st.warning("No captions generated. Audio may be silent or not in Chinese.")
                        
                        if st.button("Download Video (records stream to MP4)"):
                            with st.spinner("Downloading video stream..."):
                                output_file = f"{selected_row['title']}.mp4".replace('/', '_').replace('\\', '_')
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
            else:
                st.error("Failed to scrape metadata. Please check the URL.")
    else:
        st.warning("Please enter a valid video URL.")
else:
    st.info("Enter a video URL to get started.")
