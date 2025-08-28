# 📝 Self-Reference & Emotion Detector from Tweets

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-green?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?logo=tensorflow&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-Transformers-purple)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-orange?logo=scikit-learn&logoColor=white)
![Emoji](https://img.shields.io/badge/Emoji-🔤-yellow)

---

## 🚀 Project Overview
**Self-Reference & Emotion Detector** is a **Flask-based web application** that processes user-submitted tweets to:

- Detect if a tweet is **self-referential** using a BERT-based classifier.
- Predict the **emotion** expressed in self-referential tweets using a second BERT model.
- Display **confidence scores** for both self-reference and emotion predictions.
- Showcase **clean, emoji-free, and preprocessed** text for accurate NLP inference.

This project demonstrates **NLP**, **Deep Learning**, and **Web Development** skills in a real-world context.

---

## 🌟 Key Features

| Feature | Description | Example |
|---------|------------|--------|
| Self-Reference Detection | Identify if a tweet is about the user | Yes / No |
| Emotion Classification | Detect emotions like happiness, sadness, anger | Happy, Sad, Angry |
| Confidence Scores | Probability of predictions | 95.32% |
| Text Preprocessing | Cleans emoji, URLs, mentions, hashtags | Input → Cleaned Text |
| Interactive Web UI | Simple, user-friendly interface via Flask | Web Form |

---

## 🛠️ Technologies & Libraries

## 🛠 Libraries Used

| Library | Purpose |
|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white) **Python** | Core programming language |
| ![Flask](https://img.shields.io/badge/Flask-2.3-green?logo=flask&logoColor=white) **Flask** | Backend web framework |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?logo=tensorflow&logoColor=white) **TensorFlow** | Deep Learning model training & inference |
| ![BERT](https://img.shields.io/badge/BERT-Transformers-purple) **Transformers (BERT)** | Pretrained embeddings & NLP models |
| ![NumPy](https://img.shields.io/badge/NumPy-1.26-blue?logo=numpy&logoColor=white) **NumPy** | Numerical computations & array handling |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-teal?logo=plotly&logoColor=white) **Matplotlib** | Data visualization & plotting |
| ![Regex](https://img.shields.io/badge/Regex-Expressions-red?logo=regex&logoColor=white) **Regex** | Pattern matching & text cleaning |
| ![String](https://img.shields.io/badge/String-Utils-lightgrey?logo=python&logoColor=white) **String** | Python’s string manipulation utilities |
| ![NLTK](https://img.shields.io/badge/NLTK-3.9-green?logo=nltk&logoColor=white) **NLTK** | Tokenization, stemming, and NLP preprocessing |
| ![spaCy](https://img.shields.io/badge/spaCy-3.7-blue?logo=spacy&logoColor=white) **spaCy** | Advanced NLP processing (NER, POS tagging) |
| ![Datasketch](https://img.shields.io/badge/Datasketch-Tools-orange?logo=python&logoColor=white) **Datasketch** | Similarity search & MinHashing |
| ![LangDetect](https://img.shields.io/badge/LangDetect-🌐-purple) **LangDetect** | Detect text language automatically |
| ![scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikitlearn&logoColor=white) **Scikit-learn** | ML preprocessing & model utilities |
| ![tqdm](https://img.shields.io/badge/tqdm-Progress%20Bar-yellowgreen?logo=python&logoColor=white) **tqdm** | Progress bars for loops & training |
| ![SymSpell](https://img.shields.io/badge/SymSpell-Spell%20Correction-yellow?logo=python&logoColor=white) **SymSpell** | Fast spell checking & correction |
| ![WordCloud](https://img.shields.io/badge/WordCloud-Visualization-blueviolet?logo=wordcloud&logoColor=white) **WordCloud** | Generate word cloud visualizations |
| ![Emoji](https://img.shields.io/badge/Emoji-🔤-yellow) **Emoji** | Remove & preprocess emojis from text |

---

## 🎯 How It Works

1. **Tweet Input**  
   Enter a tweet via the Flask web interface.

2. **Text Preprocessing**  
   - Removes emojis, URLs, mentions, hashtags, numbers, and non-ASCII characters.
   - Converts multiple spaces to a single space.

3. **Self-Reference Detection**  
   - BERT tokenizer encodes the cleaned tweet.
   - TensorFlow SavedModel predicts if the tweet is self-referential.

4. **Emotion Classification (if self-referential)**  
   - Another BERT tokenizer encodes the tweet.
   - Pretrained BERT embeddings feed into the emotion model.
   - Predicts the emotion with confidence scores.

5. **Output**  
   - Displays: Tweet → Self-Reference (Yes/No) → Emotion → Confidence Scores.

---

## 🖥️ Demo

| Input Tweet | Prediction |
|------------|------------|
| "I am feeling great today!" | Self-Reference: Yes | Emotion: Happy | Confidence: 97.5% |
| "Weather is bad today" | Self-Reference: No | Emotion: Skipped | Confidence: 0% |

---

## FLOWCHART


![Screenshot_28-8-2025_20587_www canva com](https://github.com/user-attachments/assets/c582b940-b93c-4218-a679-1bcaf3a6431d)

---
![Screenshot_29-8-2025_0214_www canva com](https://github.com/user-attachments/assets/bd68fabb-78d3-4e7b-bb51-5baa158cee29)

---
![Screenshot_29-8-2025_0303_www canva com](https://github.com/user-attachments/assets/7309dfe1-c326-4166-aca2-4009553d50b9)

---





https://github.com/user-attachments/assets/ef7a4ba2-ca82-4509-ae71-5eee0e9006a3



----

## 🔮 Future Enhancements

- Add multi-language tweet support.
- Integrate social media platforms API for live tweet analysis.
- Enhance UI with more detailed visual feedback and analytics dashboards.
- Implement model retraining pipelines with user-labeled data.



## 💻 Installation


1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/self-emotion-detector.git
cd self-emotion-detector
