# 🐝 StudyBee — Intelligent Multimodal Assistant for Emotion-Aware Learning

> An adaptive learning platform that combines AI-driven document analysis, conversational assistance, and real-time emotion recognition to personalize the student experience.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Modules](#modules)
  - [Facial Emotion Recognition](#1-facial-emotion-recognition)
  - [Wearable Physiological Signal Detection](#2-wearable-physiological-signal-detection)
  - [Automatic Quiz Generation](#3-automatic-quiz-generation-from-pdfs)
  - [RAG-Based Chatbot](#4-rag-based-chatbot)
  - [Buzzy Avatar](#5-buzzy-avatar)
  - [Text-Based Emotion & Physical State Classification](#6-text-based-emotion--physical-state-classification)
  - [PPO-Based Adaptive Difficulty Controller](#7-ppo-based-adaptive-difficulty-controller)
  - [Speech Emotion Recognition](#8-speech-emotion-recognition-ravdess)
  - [Personalised Pomodoro Duration Prediction](#9-personalised-pomodoro-duration-prediction)
- [Tech Stack](#tech-stack)
- [Datasets](#datasets)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Team](#team)

---

## Overview

Modern students face increasing challenges managing time, processing complex content, and maintaining emotional well-being. Existing tools treat all learners uniformly — ignoring how a student's capacity to learn shifts depending on their emotional and cognitive state.

**StudyBee** bridges this gap by integrating:
- 📄 **Document Analysis** — PDF parsing, summarization, quiz generation
- 🤖 **Conversational AI** — RAG-based chatbot + emotionally adaptive avatar
- 😊 **Multimodal Emotion Recognition** — facial expressions, voice, physiological signals, cursor behavior, and journaling
- 🎯 **Adaptive Learning** — dynamically adjusts difficulty and Pomodoro session length based on user state

---

## Features

| Feature | Description |
|---|---|
| 📸 Facial Emotion Recognition | Real-time CNN-based detection of 7 emotions using webcam |
| ❤️ Physiological Sensing | Stress/affect detection via wearable ECG & BVP signals |
| 📝 Quiz Generation | Automatic quiz creation from uploaded PDFs using fine-tuned T5 |
| 💬 RAG Chatbot | Document-grounded Q&A with diagram and workflow generation |
| 🐝 Buzzy Avatar | Emotionally adaptive conversational agent with voice & animation |
| 🧠 Text Classification | XLM-RoBERTa-based emotion and physical state inference from text |
| 🎮 Adaptive Difficulty | PPO reinforcement learning agent for cognitive training tasks |
| 🎙️ Speech Emotion Recognition | BiLSTM + attention model on RAVDESS for 8-class speech emotion |
| ⏱️ Pomodoro Prediction | Personalised focus session duration from multimodal sensor data |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          StudyBee                               │
│                                                                 │
│  Frontend: React.js + TypeScript + Tailwind CSS                 │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │  RAG Chatbot │   │ Buzzy Avatar │   │ Adaptive Engine  │   │
│  └──────────────┘   └──────────────┘   └──────────────────┘   │
│                                                                 │
│  Backend: Django REST Framework (Python)                        │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐                          │
│  │  PostgreSQL  │   │   ChromaDB   │                          │
│  │  (relational)│   │  (vectors)   │                          │
│  └──────────────┘   └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Modules

### 1. Facial Emotion Recognition

Detects and classifies 7 facial emotions in real-time during study sessions.

- **Dataset**: FER-2013 (35,887 images, 48×48 px, 7 classes)
- **Model**: Deep CNN with 3 convolutional blocks (64 → 128 → 256 filters), BatchNorm, MaxPooling, Dropout (25%)
- **Training**: Adam optimizer, lr=0.001, up to 50 epochs with early stopping, class weighting for imbalance
- **Augmentation**: Rotation, zoom, horizontal flipping, spatial shifting

**Emotion Classes**: Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise

---

### 2. Wearable Physiological Signal Detection

Three-class affect recognition (Baseline / Stress / Amusement) from wearable sensors.

- **Dataset**: WESAD (chest ECG ~700 Hz + wrist BVP/PPG ~64 Hz)
- **Features**: BPM, SpO2 proxy, delta features, rolling means, local deviations, BPM/SpO2 ratio (9 variables total)
- **Models**: Random Forest (macro-AUC: 0.873) vs MLP (macro-AUC: 0.819)
- **Imbalance handling**: Stratified split + SMOTE (training set only) + class weighting
- **Explainability**: SHAP TreeExplainer

---

### 3. Automatic Quiz Generation from PDFs

End-to-end pipeline that converts PDF documents into interactive quizzes.

- **Model**: T5-small fine-tuned on SQuAD (up to 5,000 samples)
- **Training**: AdamW, lr=5×10⁻⁵, cosine scheduler, 3 epochs, batch size 8
- **Pipeline**:
  1. PDF text extraction with `pdfplumber`
  2. Sentence segmentation with NLTK + spaCy NER
  3. Question generation via fine-tuned T5 (beam search)
  4. Rule-based validation & quality scoring
  5. Export to JSON and HTML
- **Output**: Top-N questions with difficulty tags and relevance scores

---

### 4. RAG-Based Chatbot

Retrieval-Augmented Generation chatbot for deep academic document comprehension.

- **Stack**: PyMuPDF · Sentence Transformers (all-MiniLM-L6-v2) · ChromaDB · Groq API (LLaMA 3.3 70B) · ElevenLabs TTS · Mermaid.js
- **Pipeline**:
  1. **Ingestion**: Chunking into 500-char segments with 50-char overlap
  2. **Embedding**: 384-dim dense vectors stored in ChromaDB
  3. **Retrieval**: Cosine similarity search over session-specific collections
  4. **Generation**: Structured prompts for Q&A, summarization, diagrams, and workflows
- **Voice**: MediaRecorder API → Whisper large-v3 transcription → intent classification
- **Persistence**: All sessions stored in PostgreSQL

---

### 5. Buzzy Avatar

Emotionally intelligent conversational agent with synchronized audio-visual feedback.

- **Stack**: React.js + Framer Motion · Web Speech API · Groq API (LLaMA 3.1 8B) · Edge TTS (Azure Neural Voices)
- **Features**:
  - Real-time speech recognition (low-latency, no backend)
  - Emotion-aware response generation (JSON schema: text + emotion + language)
  - Multilingual adaptive TTS with prosodic adjustments
  - Animated avatar with lip-sync and dynamic facial expressions

---

### 6. Text-Based Emotion & Physical State Classification

Dual-task transformer classification from free-form user text input.

- **Model**: XLM-RoBERTa (multilingual, transformer-based)
- **Tasks**: Emotion detection + Physical state prediction (two independent classification heads)
- **Training**: Batch 8, lr=2×10⁻⁵, 20 epochs; weighted cross-entropy for physical state imbalance
- **Framework**: Hugging Face Transformers + Trainer API

---

### 7. PPO-Based Adaptive Difficulty Controller

Reinforcement learning agent that keeps users in the Zone of Proximal Development (ZPD).

- **Tasks**: Stroop · N-Back · Schulte · Kakuro
- **State**: 8-dimensional (accuracy, reaction time, error rates, difficulty level, session count, trend, recency)
- **Actions**: 5 discrete adjustments (−2 to +2 difficulty levels)
- **Reward**: Weighted combination of accuracy (target ~77.5%), speed, improvement trend, stability
- **Algorithm**: PPO with clipped surrogate objective, entropy regularization, gradient clipping
- **Storage**: Per-user, per-task model persistence in relational database (PyTorch serialization)

---

### 8. Speech Emotion Recognition (RAVDESS)

Frame-level sequence modeling for 8-class speech emotion classification.

- **Dataset**: RAVDESS (8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Features**: 40 MFCCs + 40 delta + 40 delta-delta → shape (130, 120) per utterance
- **Model**: Stacked BiLSTM (256→128) + Bahdanau attention + Dense(64) + Softmax(8)
- **Training**: Adam lr=0.001, up to 100 epochs, early stopping (patience=20), LR reduction on plateau
- **Explainability**: Attention visualization + SHAP KernelExplainer + LIME

---

### 9. Personalised Pomodoro Duration Prediction

Predicts optimal Pomodoro session length (15–55 min) from multimodal student sensor data.

- **Dataset**: StudentLife (Dartmouth — stress EMA, sleep logs, activity, location, mood EMA)
- **Target**: Engineered focus score from 4 pillars:
  ```
  focus_score = 0.35×sleep + 0.30×(1−stress) + 0.25×mood + 0.10×hour_peak
  ```
- **Models**: Random Forest vs XGBoost (temporal cross-validation, no data leakage)
- **Personalisation**: Adaptive blending of global + per-student models (up to 85% personal weight for students with 50+ records)
- **Clustering**: K-Means behavioral profiles — *Natural Focuser*, *Under Pressure*, *Night Owl*, *Variable Performer*
- **Explainability**: SHAP TreeExplainer (beeswarm, bar chart, waterfall plots)

---

## Tech Stack

### Backend
| Tool | Purpose |
|---|---|
| Python / Django REST Framework | API layer |
| PostgreSQL | Relational data persistence |
| ChromaDB | Vector database for semantic search |
| PyTorch | Deep learning models |
| Hugging Face Transformers | NLP models (T5, XLM-RoBERTa) |
| Groq API | LLM inference (LLaMA 3.3 70B / 3.1 8B) |
| NeuroKit2 | ECG/physiological signal processing |

### Frontend
| Tool | Purpose |
|---|---|
| React.js + TypeScript | UI framework |
| Tailwind CSS | Styling |
| Framer Motion | Avatar animations |
| Mermaid.js | Diagram rendering |
| Web Speech API | Real-time speech recognition |
| MediaRecorder API | Audio capture |

### AI / ML
| Tool | Purpose |
|---|---|
| TensorFlow / Keras | CNN, BiLSTM models |
| scikit-learn | Random Forest, preprocessing |
| XGBoost | Gradient boosting regression |
| SHAP / LIME | Model explainability |
| ElevenLabs | Neural TTS synthesis |
| Edge TTS (Azure) | Multilingual speech synthesis |
| Whisper large-v3 | Speech-to-text transcription |
| spaCy / NLTK | NLP preprocessing |

---

## Datasets

| Dataset | Task | Size |
|---|---|---|
| FER-2013 | Facial emotion recognition | 35,887 images |
| WESAD | Wearable affect detection | Chest ECG + wrist BVP |
| SQuAD | Quiz generation fine-tuning | Up to 5,000 samples used |
| RAVDESS | Speech emotion recognition | 8-class WAV recordings |
| StudentLife | Pomodoro duration prediction | Full academic term, Dartmouth |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/studybee.git
cd studybee

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Database setup
python manage.py migrate

# Run backend server
python manage.py runserver

# Frontend setup
cd ../frontend
npm install
npm run dev
```

> ⚠️ Make sure to configure your `.env` file with API keys for Groq, ElevenLabs, and any other external services before running.

---

## Project Structure

```
studybee/
├── backend/
│   ├── api/                    # Django REST endpoints
│   ├── chatbot/                # RAG pipeline & Buzzy Avatar
│   ├── emotion/                # FER, speech, text emotion modules
│   ├── quiz/                   # T5 quiz generation pipeline
│   ├── rl_agent/               # PPO difficulty controller
│   ├── pomodoro/               # Pomodoro prediction pipeline
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── pages/              # Application pages
│   │   └── assets/             # Static files & avatar PNGs
│   └── package.json
├── models/                     # Saved model artifacts
├── notebooks/                  # Research & training notebooks
└── README.md
```

---

## Team

**StudyBee Group** — May 2026

---

> *StudyBee is an academic research project exploring the intersection of affective computing and adaptive educational technology.*
