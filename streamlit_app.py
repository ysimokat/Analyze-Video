"""
Multi-Modal Video Analyzer - Streamlit Dashboard
Interactive web interface demonstrating computer vision + NLP integration
"""

import streamlit as st
import sys
from pathlib import Path
import time
import json
from datetime import datetime
import tempfile
import shutil
import cv2
import yt_dlp
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import VisualAnalyzer, TextAnalyzer, SummaryGenerator
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger, get_logger

# Page configuration
st.set_page_config(
    page_title="Multi-Modal Video Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
setup_logger(level="INFO", log_file="logs/streamlit_app.log")
logger = get_logger("StreamlitApp")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .step-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .claim-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = {
        'step': 0,
        'video_path': None,
        'frames_dir': None,
        'visual_summary': None,
        'text_summary': None,
        'report': None,
        'video_url': None
    }


def download_youtube_video(url: str, output_dir: Path) -> Optional[Path]:
    """Download YouTube video"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return Path(filename)

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def extract_frames(video_path: Path, output_dir: Path, fps: float = 0.5) -> int:
    """Extract frames from video"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = output_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        return saved_count

    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return 0


def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">🎬 Multi-Modal Video Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown("**Demonstrating Computer Vision (OCR) + Natural Language Processing Integration**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/video.png", width=80)
        st.title("⚙️ Configuration")

        # Load config
        config = get_config()

        st.markdown("### API Settings")
        api_provider = st.selectbox(
            "AI Provider",
            ["openai", "anthropic"],
            index=0
        )

        api_key_input = st.text_input(
            f"{api_provider.upper()} API Key",
            type="password",
            help="Enter your API key or set in .env file"
        )

        st.markdown("### Processing Settings")
        fps = st.slider("Frame Extraction FPS", 0.1, 2.0, 0.5, 0.1)
        ocr_confidence = st.slider("OCR Confidence Threshold", 0.3, 0.9, 0.5, 0.05)

        st.markdown("---")
        st.markdown("### About")
        st.info("""
        Multi-modal AI demonstration combining:
        - 🎬 Computer Vision (OCR)
        - 💬 Natural Language Processing
        - 🤖 LLM Integration (GPT-4)
        - 📊 Data Fusion & Analysis
        """)

        if st.button("🔄 Reset Analysis"):
            st.session_state.analysis_state = {
                'step': 0,
                'video_path': None,
                'frames_dir': None,
                'visual_summary': None,
                'text_summary': None,
                'report': None,
                'video_url': None
            }
            st.rerun()

    # Main content area
    tabs = st.tabs([
        "🎯 Step 1: Select Video",
        "🎬 Step 2: Extract Frames",
        "👁️ Step 3: Visual Analysis (OCR)",
        "💬 Step 4: Text Analysis (NLP)",
        "📊 Step 5: Multi-Modal Fusion",
        "📋 Step 6: View Report"
    ])

    # ================== STEP 1: Select Video ==================
    with tabs[0]:
        st.markdown('<div class="step-header">Step 1: Select Video File</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        Select a video file from your local system to begin analysis.
        The analyzer works with any video content - tutorials, presentations, lectures, etc.
        """)

        col1, col2 = st.columns([3, 1])

        with col1:
            video_url = st.text_input(
                "Video URL (optional - for reference only)",
                placeholder="https://example.com/video or leave blank",
                value=st.session_state.analysis_state.get('video_url', '')
            )

        # Check for available videos
        videos_dir = Path("./data/videos")
        videos_dir.mkdir(parents=True, exist_ok=True)
        available_videos = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mkv")) + list(videos_dir.glob("*.avi"))

        if available_videos:
            st.markdown("---")
            st.markdown("### 📹 Available Videos")
            st.markdown("Place video files in `data/videos/` folder")

            # Create dropdown with available videos
            video_options = {v.name: v for v in available_videos}

            # Default selection
            default_video = None
            if st.session_state.analysis_state['video_path']:
                default_video = st.session_state.analysis_state['video_path'].name
            elif available_videos:
                default_video = available_videos[0].name

            selected_video_name = st.selectbox(
                "Select a video to analyze:",
                options=list(video_options.keys()),
                index=list(video_options.keys()).index(default_video) if default_video in video_options else 0
            )

            video_path = video_options[selected_video_name]

            # Update session state
            if video_url:
                st.session_state.analysis_state['video_url'] = video_url
            st.session_state.analysis_state['video_path'] = video_path

            # Show video info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Filename", video_path.name)
            with col2:
                st.metric("Size", f"{video_path.stat().st_size / 1024 / 1024:.1f} MB")

            st.success("✅ Video selected! Continue to **Step 2** to extract frames.")
        else:
            st.warning("⚠️ No videos found in `data/videos/` folder")
            st.markdown("""
            **To add videos:**
            1. Place your video files (`.mp4`, `.mkv`, `.avi`) in the `data/videos/` folder
            2. Refresh this page
            3. Select your video from the dropdown
            """)

    # ================== STEP 2: Extract Frames ==================
    with tabs[1]:
        st.markdown('<div class="step-header">Step 2: Extract Video Frames</div>',
                    unsafe_allow_html=True)

        # Check for available videos in data/videos folder
        videos_dir = Path("./data/videos")
        videos_dir.mkdir(parents=True, exist_ok=True)
        available_videos = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mkv")) + list(videos_dir.glob("*.avi"))

        video_path = None

        if available_videos:
            st.markdown("### 📹 Available Videos")

            # Create dropdown with available videos
            video_options = {v.name: v for v in available_videos}

            # Default selection
            default_video = None
            if st.session_state.analysis_state['video_path']:
                default_video = st.session_state.analysis_state['video_path'].name
            elif available_videos:
                default_video = available_videos[0].name

            selected_video_name = st.selectbox(
                "Select a video to process:",
                options=list(video_options.keys()),
                index=list(video_options.keys()).index(default_video) if default_video in video_options else 0
            )

            video_path = video_options[selected_video_name]

            # Update session state
            st.session_state.analysis_state['video_path'] = video_path

            st.markdown(f"""
            **Selected Video:** `{video_path.name}`
            **Size:** {video_path.stat().st_size / 1024 / 1024:.2f} MB

            Frames will be extracted at **{fps} FPS** (1 frame every {1/fps:.1f} seconds).
            This provides a good balance between detail and processing time.
            """)
        elif st.session_state.analysis_state['video_path']:
            video_path = st.session_state.analysis_state['video_path']

            st.markdown(f"""
            **Video:** `{video_path.name}`

            Frames will be extracted at **{fps} FPS** (1 frame every {1/fps:.1f} seconds).
            This provides a good balance between detail and processing time.
            """)
        else:
            st.warning("⚠️ No videos found in data/videos folder. Please download a video first (Step 1)")

        if video_path:

            if st.button("🎬 Extract Frames", type="primary"):
                frames_dir = Path(f"./data/output/frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

                with st.spinner("Extracting frames from video..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("📹 Opening video file...")
                    progress_bar.progress(10)

                    status_text.text("✂️ Extracting frames...")
                    progress_bar.progress(30)

                    num_frames = extract_frames(video_path, frames_dir, fps)

                    progress_bar.progress(100)
                    status_text.text("✅ Extraction complete!")

                if num_frames > 0:
                    st.session_state.analysis_state['frames_dir'] = frames_dir
                    st.session_state.analysis_state['step'] = 2

                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success(f"✅ Extracted {num_frames} frames successfully!")
                    st.markdown(f"**Location:** `{frames_dir}`")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Show sample frames
                    st.markdown("### 📸 Sample Frames")
                    frame_files = sorted(list(frames_dir.glob("*.jpg")))

                    if len(frame_files) >= 4:
                        cols = st.columns(4)
                        indices = [0, len(frame_files)//3, 2*len(frame_files)//3, -1]
                        for idx, col in zip(indices, cols):
                            with col:
                                st.image(str(frame_files[idx]),
                                       caption=f"Frame {idx if idx >= 0 else len(frame_files)-1}",
                                       use_container_width=True)

                    st.info("👉 Continue to **Step 3** for visual analysis")
                else:
                    st.error("❌ Frame extraction failed")

            # Show current status
            if st.session_state.analysis_state['frames_dir']:
                st.markdown("---")
                st.markdown("### Extracted Frames")
                frames_dir = st.session_state.analysis_state['frames_dir']
                num_frames = len(list(frames_dir.glob("*.jpg")))
                st.info(f"📁 {num_frames} frames in `{frames_dir.name}`")

    # ================== STEP 3: Visual Analysis ==================
    with tabs[2]:
        st.markdown('<div class="step-header">Step 3: Visual Analysis with OCR</div>',
                    unsafe_allow_html=True)

        if not st.session_state.analysis_state['frames_dir']:
            st.warning("⚠️ Please extract frames first (Step 2)")
        else:
            frames_dir = st.session_state.analysis_state['frames_dir']
            num_frames = len(list(frames_dir.glob("*.jpg")))

            st.markdown(f"""
            **Frames to analyze:** {num_frames}

            Computer vision analysis will extract:
            - 📝 Text from slides and graphics (OCR)
            - 🏷️ Key terms and entities
            - 🔢 Numbers, metrics, and statistics
            - 🎯 Key visual moments
            """)

            if st.button("🔍 Analyze Visual Content", type="primary"):
                with st.spinner("Initializing Visual Analyzer (may take a minute on first run)..."):
                    visual_analyzer = VisualAnalyzer()

                st.success("✅ Analyzer initialized!")

                with st.spinner(f"Analyzing {num_frames} frames... This may take several minutes."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("🔍 Running OCR on frames...")
                    progress_bar.progress(20)

                    start_time = time.time()
                    visual_summary = visual_analyzer.analyze_frames(
                        frames_dir=frames_dir,
                        batch_size=16,
                        confidence_threshold=ocr_confidence
                    )
                    analysis_time = time.time() - start_time

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")

                st.session_state.analysis_state['visual_summary'] = visual_summary
                st.session_state.analysis_state['step'] = 3

                # Display results
                st.markdown("---")
                st.markdown("### 📊 Visual Analysis Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Frames", visual_summary.total_frames)

                with col2:
                    st.metric("Frames with Text", visual_summary.frames_with_text)

                with col3:
                    st.metric("Key Terms Found", len(visual_summary.detected_tickers))

                with col4:
                    st.metric("Data Points", len(visual_summary.numbers_found))

                st.markdown(f"**Analysis Time:** {analysis_time:.1f} seconds")

                # Detected Key Terms
                if visual_summary.detected_tickers:
                    st.markdown("### 🏷️ Detected Key Terms")
                    ticker_cols = st.columns(min(len(visual_summary.detected_tickers), 5))
                    for idx, ticker in enumerate(visual_summary.detected_tickers[:5]):
                        with ticker_cols[idx]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3 style="color: #1f77b4; margin: 0;">{ticker}</h3>
                            </div>
                            """, unsafe_allow_html=True)

                    if len(visual_summary.detected_tickers) > 5:
                        with st.expander(f"Show all {len(visual_summary.detected_tickers)} terms"):
                            st.write(", ".join(visual_summary.detected_tickers))

                # Numbers Found
                if visual_summary.numbers_found:
                    st.markdown("### 📈 Extracted Data Points")
                    for i, num in enumerate(visual_summary.numbers_found[:10]):
                        st.markdown(f"""
                        <div class="claim-card">
                            <strong>Value:</strong> {num.get('value', 'N/A')} &nbsp;&nbsp;
                            <strong>Confidence:</strong> {num.get('confidence', 0):.0%}
                        </div>
                        """, unsafe_allow_html=True)

                    if len(visual_summary.numbers_found) > 10:
                        st.info(f"+ {len(visual_summary.numbers_found) - 10} more data points")

                # Key Frames
                if visual_summary.key_frames:
                    st.markdown("### 🔑 Key Frames Identified")
                    st.write(f"Frames with significant visual content: {visual_summary.key_frames[:10]}")

                st.info("👉 Continue to **Step 4** for text analysis")

            # Show current status
            if st.session_state.analysis_state['visual_summary']:
                st.markdown("---")
                st.markdown("### Analysis Complete")
                vs = st.session_state.analysis_state['visual_summary']
                st.success(f"✅ Found {len(vs.detected_tickers)} key terms and {len(vs.numbers_found)} data points")

    # ================== STEP 4: Text Analysis ==================
    with tabs[3]:
        st.markdown('<div class="step-header">Step 4: Text Analysis (NLP)</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        **Audio Transcription + AI Analysis**

        In a production system, audio would be transcribed using OpenAI Whisper.
        For this demo, you can paste a transcript or use the sample below.
        """)

        # Transcript input
        use_sample = st.checkbox("Use sample transcript", value=True)

        if use_sample:
            transcript = """
            Hello everyone, welcome to this tutorial on Python programming. Today we'll cover several important concepts
            including data structures, functions, and best practices.

            First, let's talk about lists and dictionaries. Lists are ordered collections that can store multiple items.
            For example, you might have a list of numbers like [1, 2, 3, 4, 5]. Dictionaries, on the other hand, store
            key-value pairs, which is useful for organizing related data.

            Next, I'll show you how to write clean functions. A good function should do one thing well. We'll use the
            'def' keyword to define functions, and remember to include docstrings explaining what each function does.
            This makes your code more maintainable.

            Error handling is crucial in Python. We use try-except blocks to handle potential errors gracefully.
            This prevents your program from crashing when something unexpected happens. For instance, when reading files
            or making network requests, always include proper error handling.

            Now let's discuss performance. Using list comprehensions instead of traditional loops can make your code
            faster and more readable. The same goes for generator expressions when working with large datasets.

            In conclusion, writing clean, maintainable Python code involves understanding data structures, writing
            focused functions, handling errors properly, and optimizing where necessary. Practice these concepts
            and you'll become a better programmer.
            """
            st.text_area("Transcript", transcript, height=200, disabled=True)
        else:
            transcript = st.text_area(
                "Paste your transcript here",
                height=200,
                placeholder="Paste the video transcript here..."
            )

        if st.button("🤖 Analyze with AI", type="primary", disabled=not transcript):
            # Check API key
            if not api_key_input:
                try:
                    config = get_config()
                    api_key = config.get_api_key(api_provider)
                except:
                    st.error("❌ API key not found. Please enter it in the sidebar or set in .env file.")
                    st.stop()

            with st.spinner("Initializing AI Text Analyzer..."):
                try:
                    text_analyzer = TextAnalyzer()
                    st.success("✅ Analyzer initialized!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {e}")
                    st.stop()

            with st.spinner("Analyzing transcript with AI... This may take 30-60 seconds."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("📝 Generating summary...")
                    progress_bar.progress(20)

                    status_text.text("🔍 Extracting claims...")
                    progress_bar.progress(40)

                    status_text.text("💭 Analyzing content tone...")
                    progress_bar.progress(60)

                    status_text.text("🏢 Identifying entities...")
                    progress_bar.progress(80)

                    start_time = time.time()
                    text_summary = text_analyzer.analyze_transcript(transcript)
                    analysis_time = time.time() - start_time

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")

                    st.session_state.analysis_state['text_summary'] = text_summary
                    st.session_state.analysis_state['step'] = 4

                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Text Analysis Results")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Content Tone", text_summary.sentiment.value.upper())

                    with col2:
                        st.metric("Claims Found", len(text_summary.claims))

                    with col3:
                        st.metric("Entities", len(text_summary.mentioned_entities))

                    with col4:
                        st.metric("Topics", len(text_summary.topics))

                    st.markdown(f"**Analysis Time:** {analysis_time:.1f} seconds")

                    # Executive Summary
                    st.markdown("### 📝 Executive Summary")
                    st.markdown(f"""
                    <div class="info-box">
                    {text_summary.executive_summary}
                    </div>
                    """, unsafe_allow_html=True)

                    # Key Points
                    if text_summary.key_points:
                        st.markdown("### 📌 Key Points")
                        for i, point in enumerate(text_summary.key_points, 1):
                            st.markdown(f"{i}. {point}")

                    # Mentioned Entities
                    if text_summary.mentioned_entities:
                        st.markdown("### 🏷️ Mentioned Entities")
                        entity_cols = st.columns(min(len(text_summary.mentioned_entities), 5))
                        for idx, entity in enumerate(text_summary.mentioned_entities[:5]):
                            with entity_cols[idx]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin: 0;">{entity}</h4>
                                </div>
                                """, unsafe_allow_html=True)

                    # Claims
                    if text_summary.claims:
                        st.markdown("### 🔍 Extracted Claims")

                        # Filter by type
                        claim_types = list(set([c.type.value for c in text_summary.claims]))
                        selected_type = st.selectbox("Filter by type", ["All"] + claim_types)

                        filtered_claims = text_summary.claims
                        if selected_type != "All":
                            filtered_claims = [c for c in text_summary.claims if c.type.value == selected_type]

                        for claim in filtered_claims[:10]:
                            confidence_color = "#28a745" if claim.confidence > 0.7 else "#ffc107" if claim.confidence > 0.5 else "#dc3545"

                            st.markdown(f"""
                            <div class="claim-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span><strong>Type:</strong> <span style="color: #1f77b4;">{claim.type.value}</span></span>
                                    <span><strong>Confidence:</strong> <span style="color: {confidence_color};">{claim.confidence:.0%}</span></span>
                                </div>
                                <p style="margin: 0.5rem 0;"><strong>Claim:</strong> {claim.text}</p>
                                {f'<p style="margin: 0;"><strong>Related:</strong> {", ".join(claim.entities)}</p>' if claim.entities else ''}
                            </div>
                            """, unsafe_allow_html=True)

                        if len(filtered_claims) > 10:
                            st.info(f"+ {len(filtered_claims) - 10} more claims")

                    # Topics
                    if text_summary.topics:
                        st.markdown("### 🏷️ Topics Discussed")
                        st.write(", ".join(text_summary.topics))

                    st.info("👉 Continue to **Step 5** to generate the final report")

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    logger.error(f"Text analysis error: {e}", exc_info=True)

        # Show current status
        if st.session_state.analysis_state['text_summary']:
            st.markdown("---")
            st.markdown("### Analysis Complete")
            ts = st.session_state.analysis_state['text_summary']
            st.success(f"✅ Extracted {len(ts.claims)} claims with {ts.sentiment.value} tone")

    # ================== STEP 5: Generate Report ==================
    with tabs[4]:
        st.markdown('<div class="step-header">Step 5: Multi-Modal Fusion</div>',
                    unsafe_allow_html=True)

        if not st.session_state.analysis_state.get('text_summary'):
            st.warning("⚠️ Please complete text analysis first (Step 4)")
        else:
            st.markdown("""
            **Combine Computer Vision + NLP Results**

            This step fuses visual and text analysis to generate:
            - 📊 Complete content analysis
            - 🎯 Quality scoring
            - 📋 Pattern detection
            - 💡 Key insights
            """)

            if st.button("📊 Generate Report", type="primary"):
                with st.spinner("Generating comprehensive report..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("🔄 Fusing visual + text analysis...")
                    progress_bar.progress(30)

                    summary_gen = SummaryGenerator()

                    status_text.text("📊 Calculating quality score...")
                    progress_bar.progress(60)

                    status_text.text("⚠️ Identifying risk indicators...")
                    progress_bar.progress(80)

                    visual_summary = st.session_state.analysis_state.get('visual_summary')
                    text_summary = st.session_state.analysis_state['text_summary']
                    video_url = st.session_state.analysis_state.get('video_url', 'Unknown')

                    # Use mock visual summary if not available
                    if not visual_summary:
                        from src.analyzers.visual_analyzer import VisualSummary
                        visual_summary = VisualSummary(
                            total_frames=0,
                            frames_with_text=0,
                            ocr_results=[],
                            detected_tickers=[],
                            numbers_found=[],
                            key_frames=[]
                        )

                    report = summary_gen.generate_report(
                        video_url=video_url,
                        visual_summary=visual_summary,
                        text_summary=text_summary,
                        processing_time=0
                    )

                    progress_bar.progress(100)
                    status_text.text("✅ Report generated!")

                    st.session_state.analysis_state['report'] = report
                    st.session_state.analysis_state['step'] = 5

                    st.success("✅ Report generated successfully!")
                    st.info("👉 View the complete report in the **Final Report** tab")

        # Show current status
        if st.session_state.analysis_state.get('report'):
            st.markdown("---")
            st.markdown("### Report Ready")
            report = st.session_state.analysis_state['report']
            st.success(f"✅ Quality Score: {report.credibility_score:.1f}/100")

    # ================== FINAL REPORT ==================
    with tabs[5]:
        st.markdown('<div class="step-header">📋 Final Analysis Report</div>',
                    unsafe_allow_html=True)

        if not st.session_state.analysis_state.get('report'):
            st.warning("⚠️ Please generate the report first (Step 5)")
        else:
            report = st.session_state.analysis_state['report']

            # Header metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                score_color = "🟢" if report.credibility_score >= 70 else "🟡" if report.credibility_score >= 50 else "🔴"
                st.metric("Quality Score", f"{score_color} {report.credibility_score:.1f}/100")

            with col2:
                sentiment_emoji = "📈" if report.sentiment == "bullish" else "📉" if report.sentiment == "bearish" else "➡️"
                st.metric("Content Tone", f"{sentiment_emoji} {report.sentiment.upper()}")

            with col3:
                st.metric("Total Claims", len(report.factual_claims) + len(report.predictions))

            with col4:
                st.metric("Entities Found", len(report.all_mentioned_entities))

            st.markdown("---")

            # Executive Summary
            st.markdown("### 📝 Executive Summary")
            st.markdown(f"""
            <div class="info-box">
            {report.executive_summary}
            </div>
            """, unsafe_allow_html=True)

            # Key Insights
            st.markdown("### 💡 Key Insights")
            for i, insight in enumerate(report.key_insights, 1):
                st.markdown(f"{i}. {insight}")

            # Mentioned Entities
            if report.all_mentioned_entities:
                st.markdown("### 🏷️ Mentioned Entities")
                entity_text = ", ".join(report.all_mentioned_entities)
                st.markdown(f"""
                <div class="metric-card">
                {entity_text}
                </div>
                """, unsafe_allow_html=True)

            # Claims Analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ✅ Factual Claims")
                if report.factual_claims:
                    for claim in report.factual_claims[:5]:
                        st.markdown(f"""
                        <div class="claim-card">
                            <p style="margin: 0;"><strong>{claim['text']}</strong></p>
                            <small>Confidence: {claim['confidence']:.0%}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    if len(report.factual_claims) > 5:
                        st.info(f"+ {len(report.factual_claims) - 5} more factual claims")
                else:
                    st.info("No factual claims extracted")

            with col2:
                st.markdown("### 🔮 Predictions")
                if report.predictions:
                    for pred in report.predictions[:5]:
                        st.markdown(f"""
                        <div class="claim-card">
                            <p style="margin: 0;"><strong>{pred['text']}</strong></p>
                            <small>Confidence: {pred['confidence']:.0%}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    if len(report.predictions) > 5:
                        st.info(f"+ {len(report.predictions) - 5} more predictions")
                else:
                    st.info("No predictions extracted")

            # Risk Indicators
            st.markdown("### ⚠️ Risk Indicators")
            if report.risk_indicators:
                for risk in report.risk_indicators:
                    st.markdown(f"""
                    <div class="warning-box">
                    ⚠️ {risk}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                ✅ No major risk indicators detected
                </div>
                """, unsafe_allow_html=True)

            # Visual Analysis Summary
            st.markdown("### 📊 Visual Analysis Summary")
            vcol1, vcol2, vcol3, vcol4 = st.columns(4)
            with vcol1:
                st.metric("Frames Analyzed", report.visual_summary['total_frames'])
            with vcol2:
                st.metric("Frames with Text", report.visual_summary['frames_with_text'])
            with vcol3:
                st.metric("Data Points", len(report.visual_summary['numbers_found']))
            with vcol4:
                st.metric("Key Moments", len(report.key_frames))

            # Topics
            if report.topics:
                st.markdown("### 🏷️ Topics Discussed")
                st.write(", ".join(report.topics))

            # Metadata
            st.markdown("---")
            st.markdown("### 📋 Report Metadata")
            st.markdown(f"""
            - **Video URL:** {report.video_url}
            - **Analyzed:** {report.analyzed_at}
            - **Processing Time:** {report.processing_time:.1f}s
            - **Verification Status:** {report.verification_status}
            """)

            # Download buttons
            st.markdown("---")
            st.markdown("### 💾 Download Report")

            col1, col2 = st.columns(2)

            with col1:
                # JSON download
                json_str = report.to_json()
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"video_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col2:
                # Markdown download
                summary_gen = SummaryGenerator()
                md_content = summary_gen._generate_markdown(report)
                st.download_button(
                    label="📥 Download Markdown",
                    data=md_content,
                    file_name=f"video_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            # Disclaimer
            st.markdown("---")
            st.markdown("""
            <div style="background-color: #f8d7da; padding: 1rem; border-radius: 0.25rem; border-left: 4px solid #dc3545;">
            <strong>⚠️ DISCLAIMER</strong><br>
            This is an AI-generated analysis for <strong>informational purposes ONLY</strong>.
            It is NOT financial advice. Always verify claims independently and consult with
            subject matter experts when making decisions based on this content.
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
