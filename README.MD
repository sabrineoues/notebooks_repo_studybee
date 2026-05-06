# Emotion & Physical State Classification (NLP)

##  Overview
This project implements an **end-to-end NLP pipeline** to classify both **emotion** and **physical state** from user text input.

The system leverages **XLM-RoBERTa**, a pretrained multilingual transformer model, and is fine-tuned on a custom dataset containing English and French text.

---

## Objectives

###  Business Objective
Empower users through adaptive, personalized experiences by analyzing their text to understand emotional and physical states and dynamically adapting the interface.

### Technical Objective
Build a complete NLP pipeline — from preprocessing to deployment — that performs dual classification (emotion + physical state) and supports real-time inference.

---

##  Model Architecture

- **Base model:** XLM-RoBERTa (multilingual transformer)
- **Emotion model:** fine-tuned for emotion classification
- **Physical model:** fine-tuned with **weighted loss** to handle class imbalance

👉 Transfer learning is used to leverage pretrained knowledge and improve performance.

---

## Dataset

- ~250 text samples
- Multilingual (English & French)
- Labels:
  - Emotion (e.g., joy, sadness, anger…)
  - Physical state (e.g., tired, energetic…)

---

## Pipeline Overview

1. Data preprocessing  
   - Cleaning  
   - Label encoding  
   - Tokenization (XLM-RoBERTa)  
   - Train / validation / test split  

2. Modelling  
   - Two parallel models (emotion & physical)  
   - Fine-tuning of pretrained transformer  
   - Classification head + softmax  

3. Training  
   - Epochs: 15  
   - Batch size: 8  
   - Learning rate: 2e-5  
   - Optimizer: AdamW  

4. Evaluation  
   - Accuracy  
   - F1-score  
   - Precision / Recall  

5. Deployment  
   - FastAPI backend  
   - Models hosted on Hugging Face  
   - Real-time inference  

---

## Results

Example performance:

- Accuracy: ~88%
- F1-score: ~88%

👉 The model demonstrates good generalization on unseen data.

---

## 📂 Project Structure
emotion-physical-classifier/
│
├── datasets/
├── notebooks/ # Training notebooks
├── emotion_model/
├── physical_model/
├── requirements.txt
├── .gitignore
├── README.md

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/emotion-physical-classifier.git
cd emotion-physical-classifier
pip install -r requirements.txt

## 👨‍💻 Author

- Ahmed Abdallah 