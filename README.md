# 🎬 Multi-Modal Video Analyzer

> Combining Computer Vision and Natural Language Processing to extract comprehensive insights from video content.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7%2B-orange)
![GPT-4](https://img.shields.io/badge/GPT--4-OpenAI-blueviolet)

## 📋 Overview

The Multi-Modal Video Analyzer is an AI-powered system that demonstrates how to combine **computer vision** and **natural language processing** to analyze video content. It extracts text from video frames using OCR, analyzes transcripts using GPT-4, and fuses both analyses to create comprehensive insights.

**This project demonstrates:**
- Multi-modal AI architecture (vision + language)
- Transfer learning with pre-trained models
- Prompt engineering for structured output
- Data fusion across modalities
- Production-quality pipeline design

---

## ✨ Features

### 🎥 Video Processing
- **Frame Extraction**: Intelligent sampling using OpenCV (configurable FPS)
- **Format Support**: Works with MP4, MKV, AVI, and other standard formats
- **Efficient Processing**: Downsampling reduces data 60x while preserving key information

### 👁️ Computer Vision (OCR)
- **Deep Learning OCR**: Uses EasyOCR with CRAFT detection + recognition networks
- **Text Detection**: Automatically finds and reads text in video frames
- **Structured Data Extraction**: Identifies key terms, numbers, percentages, prices
- **Confidence Filtering**: Adjustable threshold to balance accuracy vs coverage

### 🧠 Natural Language Processing
- **GPT-4 Integration**: Advanced language understanding
- **Claim Extraction**: Categorizes statements as factual, prediction, or opinion
- **Entity Recognition**: Identifies mentioned topics, tools, concepts
- **Tone Analysis**: Determines overall sentiment (positive/negative/neutral)
- **Automatic Summarization**: Generates concise content overviews

### 🔀 Multi-Modal Fusion
- **Cross-Modal Correlation**: Identifies entities confirmed across both vision and text
- **Quality Scoring**: Multi-criteria evaluation (visual evidence + claim quality + completeness)
- **Insight Generation**: Creates emergent insights from combined data
- **Unified Report**: Integrates visual and textual analysis

### 📊 Output Formats
- **JSON**: Machine-readable structured data for programmatic processing
- **Markdown**: Human-readable reports for documentation and sharing
- **Interactive Dashboard**: Streamlit web interface (optional)
- **Jupyter Notebook**: Self-contained demo with inline visualizations

---

## 🏗️ Architecture

```
┌─────────────┐
│ Video Input │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│  Frame Extraction    │ ◄── OpenCV
│  (0.5 FPS sampling)  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Visual Analysis     │ ◄── EasyOCR (CRAFT + Recognition)
│  (OCR on frames)     │
└──────┬───────────────┘
       │
       ├─────────┐
       │         │
       ▼         ▼
   ┌─────┐  ┌──────────────────┐
   │Terms│  │ Text Analysis    │ ◄── GPT-4
   │ &   │  │ (NLP on          │
   │Data │  │  transcript)     │
   └──┬──┘  └────┬─────────────┘
      │          │
      │  ┌───────┴────────┐
      │  │   Entities &   │
      │  │    Claims      │
      │  └───────┬────────┘
      │          │
      └────┬─────┘
           │
           ▼
    ┌──────────────────┐
    │ Multi-Modal      │
    │ Fusion           │
    │ - Correlation    │
    │ - Scoring        │
    │ - Insights       │
    └────┬─────────────┘
         │
         ▼
    ┌──────────────────┐
    │ Final Report     │
    │ (JSON + MD)      │
    └──────────────────┘
```

### Pipeline Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frame Extraction** | OpenCV | Sample video frames at intervals |
| **Visual Analysis** | EasyOCR | Extract text from images via deep learning |
| **Text Analysis** | GPT-4 | Understand content via language models |
| **Data Fusion** | Custom Logic | Combine and correlate multi-modal data |
| **Reporting** | Python | Generate structured outputs |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT-4 access)
- 4GB+ RAM (8GB recommended for faster OCR)
- (Optional) CUDA-compatible GPU for faster processing

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd video2text2summary
```

2. **Create virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Usage

#### Option 1: Jupyter Notebook (Recommended for Learning)

```bash
jupyter notebook Multi_Modal_Video_Analyzer_Demo.ipynb
```

Run cells sequentially and follow the inline explanations.

#### Option 2: Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Access the web interface at `http://localhost:8501`

#### Option 3: Python Script

```python
from pathlib import Path
from src.video_processor import VideoProcessor
from src.analyzers.visual_analyzer import VisualAnalyzer
from src.analyzers.text_analyzer import TextAnalyzer

# Initialize
video_path = Path("data/videos/your_video.mp4")

# Extract frames
processor = VideoProcessor()
frames = processor.extract_frames(video_path, fps=0.5)

# Analyze
visual_analyzer = VisualAnalyzer()
visual_summary = visual_analyzer.analyze_frames(frames)

text_analyzer = TextAnalyzer()
text_summary = text_analyzer.analyze_transcript(transcript)

# Fuse and report
# ... (see demo notebook for complete example)
```

---

## 📁 Project Structure

```
video2text2summary/
│
├── 📓 Multi_Modal_Video_Analyzer_Demo.ipynb  # Self-contained demo
├── 🎨 streamlit_app.py                       # Web dashboard
├── 📋 requirements.txt                        # Python dependencies
├── 📄 .env.example                            # Environment variables template
│
├── 📚 src/                                    # Source code
│   ├── video_processor.py                    # Frame extraction
│   ├── analyzers/
│   │   ├── visual_analyzer.py               # OCR pipeline
│   │   └── text_analyzer.py                 # NLP pipeline
│   └── fusion/
│       └── multimodal_fusion.py             # Data fusion logic
│
├── 📂 data/                                   # Data directory
│   ├── videos/                               # Input videos
│   └── output/
│       ├── frames/                           # Extracted frames
│       └── reports/                          # Analysis reports
│
├── 📖 docs/                                   # Documentation
│   ├── INTERACTIVE_TECH_DEMO_SCRIPT.md       # Detailed demo walkthrough
│   ├── GENERAL_DEMO_SCRIPT.md                # Generic presentation script
│   ├── CONCEPTS_EXPLAINED.md                 # Beginner-friendly explanations
│   └── JUPYTER_NOTEBOOK_DEMO_SCRIPT.md       # Notebook demo guide
│
└── 🧪 tests/                                  # Unit tests (if applicable)
```

---

## 🎓 How It Works

### 1. Frame Extraction
- Opens video with OpenCV
- Samples frames at specified FPS (default: 0.5 = 1 frame every 2 seconds)
- Saves frames as JPEG images for processing

**Why downsampling?** A 5-minute video at 30 FPS = 9,000 frames. At 0.5 FPS = 150 frames (60x reduction) while capturing all important visual changes.

### 2. Visual Analysis (OCR)

EasyOCR uses two neural networks:

**CRAFT (Character Region Awareness For Text detection):**
- Scans the image to find WHERE text exists
- Returns bounding boxes around text regions

**Recognition Network:**
- Takes each text region and reads WHAT it says
- Character-level recognition with confidence scores

**Post-processing:**
- Filters detections by confidence threshold
- Extracts key terms using regex patterns
- Identifies structured data (numbers, percentages, prices)

### 3. Text Analysis (NLP)

GPT-4 processes the transcript through **prompt engineering**:

```python
prompt = """
Extract all specific claims from this content.

For each claim, provide:
- claim_text: The actual claim
- claim_type: "factual", "prediction", or "opinion"
- confidence: How verifiable (0-100)

Return as a JSON array.
"""
```

This produces structured output:
```json
[
  {
    "claim_text": "Python was created in 1991",
    "claim_type": "factual",
    "confidence": 95
  },
  {
    "claim_text": "Python is the best language",
    "claim_type": "opinion",
    "confidence": 30
  }
]
```

### 4. Multi-Modal Fusion

**Cross-modal correlation:**
```python
visual_terms = {"Python", "NumPy", "pandas"}
text_entities = {"Python", "machine learning", "NumPy"}
overlap = visual_terms ∩ text_entities  # {"Python", "NumPy"}
```

If an entity appears in BOTH video frames AND transcript, it's highly likely to be a key topic.

**Multi-criteria quality scoring:**
- Visual evidence (30%): Text density in frames
- Claim quality (40%): Ratio of factual to opinion claims
- Completeness (30%): Presence of summary, entities, key points

**Quality Score = Visual + Claim + Completeness (max 100)**

---

## 🎯 Use Cases

### Educational
- **Tutorial Analysis**: Extract key concepts from coding tutorials
- **Lecture Summarization**: Generate study guides from recorded lectures
- **Content Comparison**: Analyze explanations across multiple videos

### Business
- **Meeting Analysis**: Extract action items from recorded meetings
- **Presentation Review**: Summarize key points from presentations
- **Training Assessment**: Evaluate training video content quality

### Research
- **Content Analysis**: Systematic video analysis for research
- **Dataset Creation**: Build multi-modal datasets
- **Information Extraction**: Extract structured data from unstructured videos

### Personal
- **Video Organization**: Make personal video libraries searchable
- **Recipe Extraction**: Pull recipes from cooking videos
- **Conference Notes**: Summarize talks and presentations

---

## 🛠️ Configuration

### Frame Extraction Settings

```python
# In notebook or script
extract_frames(
    video_path=video_path,
    output_dir=FRAMES_DIR,
    fps=0.5  # Adjust sampling rate
)

# Lower FPS = fewer frames, faster processing
# Higher FPS = more frames, better temporal coverage
```

### OCR Settings

```python
analyze_frames_with_ocr(
    frames_dir=FRAMES_DIR,
    confidence_threshold=0.3  # Adjust confidence filter
)

# Lower threshold (0.2) = catch more text, more false positives
# Higher threshold (0.5) = fewer detections, higher accuracy
```

### GPT-4 Settings

```python
response = client.chat.completions.create(
    model="gpt-4",           # Use "gpt-3.5-turbo" for faster/cheaper
    temperature=0.3,         # Lower = more deterministic
    max_tokens=1000          # Adjust response length
)
```

---

## 📊 Sample Output

### JSON Report Structure
```json
{
  "video_name": "tutorial_video.mp4",
  "timestamp": "2025-01-16T10:30:00",
  "quality_score": 78.5,
  "visual_summary": {
    "total_frames": 150,
    "frames_with_text": 87,
    "detected_terms": ["Python", "NumPy", "pandas"],
    "numbers_found": [
      {"value": "95%", "numeric": 95, "is_percentage": true}
    ]
  },
  "text_summary": {
    "tone": "POSITIVE",
    "summary": "Tutorial covers Python data structures and best practices...",
    "entities": ["Python", "lists", "dictionaries", "functions"],
    "claims": [
      {
        "claim_text": "Lists are ordered collections",
        "claim_type": "factual",
        "confidence": 90
      }
    ]
  },
  "all_entities": ["Python", "NumPy", "pandas", "lists", "dictionaries"],
  "insights": [
    "Found 15 entities confirmed across both visual and text analysis",
    "Visual analysis found 12 data points",
    "Extracted 8 claims (6 factual, 2 opinions)"
  ]
}
```

---

## 🔬 Technical Concepts Demonstrated

### 1. Transfer Learning
Using pre-trained models (EasyOCR, GPT-4) instead of training from scratch. Features learned on general datasets apply to our specific use case.

### 2. Prompt Engineering
Crafting instructions to extract structured output from language models without retraining.

### 3. Multi-Modal AI
Combining different data types (vision + language) for richer insights than single-modal analysis.

### 4. Pipeline Architecture
Modular design with separation of concerns - each component does ONE thing well.

### 5. Data Fusion
Intelligently combining multiple data sources through correlation and aggregation.

---

## 📚 Documentation

- **[Interactive Tech Demo Script](docs/INTERACTIVE_TECH_DEMO_SCRIPT.md)** - Detailed 22-minute walkthrough with cell-by-cell explanations
- **[General Demo Script](docs/GENERAL_DEMO_SCRIPT.md)** - Generic presentation script for any video type
- **[Concepts Explained](docs/CONCEPTS_EXPLAINED.md)** - Beginner-friendly guide explaining WHY each technique is used
- **[Notebook Demo Script](docs/JUPYTER_NOTEBOOK_DEMO_SCRIPT.md)** - Guide for presenting the Jupyter notebook

---

## 🤝 Contributing

This is an educational project demonstrating AI/ML techniques. Feel free to:

- Fork the repository
- Experiment with different models (Tesseract, Claude, etc.)
- Add new analysis features
- Improve the fusion algorithm
- Create new demo scripts

---

## ⚖️ License

This project is for educational purposes.

**Third-party components:**
- OpenCV: Apache 2.0 License
- EasyOCR: Apache 2.0 License
- OpenAI GPT-4: Subject to OpenAI Terms of Service

---

## 🐛 Troubleshooting

### Common Issues

**1. "IProgress not found" error in Jupyter:**
```bash
pip install ipywidgets
# Restart Jupyter kernel
```

**2. OCR not detecting text:**
- Lower confidence threshold to 0.2
- Check frame quality (should be clear, high-contrast)
- Ensure frames were extracted successfully

**3. OpenAI API errors:**
- Verify API key is set in `.env`
- Check API quota and billing
- Try GPT-3.5-turbo for testing (cheaper)

**4. Out of memory errors:**
- Reduce FPS (extract fewer frames)
- Process videos in smaller chunks
- Close other applications

**5. Slow processing:**
- Enable GPU for EasyOCR (set `gpu=True`)
- Reduce number of frames
- Use parallel processing for multiple videos

---

## 📈 Performance

**Typical Processing Times** (5-minute video, CPU):

| Stage | Time | Notes |
|-------|------|-------|
| Frame Extraction | ~30 seconds | Depends on video resolution |
| OCR Analysis | ~2-5 minutes | 150 frames × 1-2 seconds each |
| GPT-4 Analysis | ~30 seconds | 5 API calls |
| Fusion & Report | ~5 seconds | Lightweight processing |
| **Total** | **~3-6 minutes** | For 5-minute video |

**With GPU:** OCR can be 3-5x faster.

---

## 🎓 Learning Resources

**Understanding the Code:**
1. Start with `CONCEPTS_EXPLAINED.md` - Learn WHY each technique is used
2. Open `Multi_Modal_Video_Analyzer_Demo.ipynb` - See everything in action
3. Follow `INTERACTIVE_TECH_DEMO_SCRIPT.md` - Detailed walkthrough

**Key Concepts to Study:**
- Computer Vision: Image processing, CNNs, OCR
- NLP: Language models, transformers, prompt engineering
- Software Engineering: Pipeline architecture, modularity, type hints
- Multi-Modal AI: Data fusion, cross-modal learning

**External Resources:**
- [OpenCV Documentation](https://docs.opencv.org/)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Multi-Modal Learning Papers](https://paperswithcode.com/task/multimodal-learning)

---

## 🌟 Acknowledgments

- **OpenCV Team** - Industry-standard computer vision library
- **JaidedAI** - EasyOCR deep learning OCR
- **OpenAI** - GPT-4 language model
- **Streamlit** - Interactive web app framework

---

## 📬 Contact

For questions, suggestions, or collaboration:
- Open an issue in this repository
- Check existing documentation in `/docs`
- Review the code comments and docstrings

---

## 🎯 Future Enhancements

Potential improvements and extensions:

- [ ] Whisper AI integration for automatic transcription
- [ ] Support for multiple languages (EasyOCR supports 80+)
- [ ] Video embedding search (find similar videos)
- [ ] Real-time processing for live streams
- [ ] Batch processing for video libraries
- [ ] Advanced fusion algorithms (attention mechanisms)
- [ ] API deployment (FastAPI/Flask)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)

---

**Built with ❤️ to demonstrate production-quality multi-modal AI architecture.**

---

## 🚀 Quick Links

- [📓 Demo Notebook](Multi_Modal_Video_Analyzer_Demo.ipynb)
- [🎨 Streamlit Dashboard](streamlit_app.py)
- [📖 Full Documentation](docs/)
- [🎓 Concepts Guide](docs/CONCEPTS_EXPLAINED.md)
- [🎬 Demo Scripts](docs/INTERACTIVE_TECH_DEMO_SCRIPT.md)
