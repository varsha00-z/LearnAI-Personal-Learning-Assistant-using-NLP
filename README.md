# LearnAI — Personal Learning Assistant using NLP

> A beginner-friendly, fully offline AI-powered study tool built with Python, NLP, and Machine Learning.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 💬 **AI Study Chatbot** | Answers student questions using TF-IDF cosine similarity retrieval |
| 📄 **Upload Notes** | Upload PDF or TXT files for instant NLP analysis |
| 📝 **Smart Quiz** | Auto-generates quizzes from the knowledge base or your own notes |
| 📊 **Progress Dashboard** | Visual analytics: scores over time, topic mastery, trend lines |
| 🔍 **NLP Explorer** | Live tokenisation, lemmatisation, TF-IDF, topic classification |
| 📌 **Weak Topic Recommendations** | AI identifies which topics need revision |

---

## 🛠️ Tech Stack

| Library | Usage |
|---|---|
| `streamlit` | Web frontend / UI |
| `nltk` | Tokenisation, stop-words, lemmatisation, POS tagging |
| `scikit-learn` | TF-IDF vectorisation, Naive Bayes classifier, cosine similarity |
| `pandas` | Knowledge base, progress data, statistics |
| `numpy` | Numerical operations, array scoring |
| `matplotlib` | Charts: bar, line, pie |
| `PyPDF2` | PDF text extraction |
| `joblib` | ML model serialisation |
| `spacy` | Available for extended NLP (installed) |

---

## 📁 Project Structure

```
LearnAI/
├── app.py                   # Main Streamlit application
├── train_model.py           # One-time model training script
├── requirements.txt         # Python dependencies
├── README.md
├── REPORT.md                # Project report
│
├── src/
│   ├── nlp_engine.py        # TextPreprocessor, TFIDFEngine, TopicClassifier
│   ├── chatbot.py           # StudyChatbot (retrieval-based)
│   ├── summarizer.py        # Extractive summariser
│   ├── quiz_generator.py    # MCQ + short-answer quiz generation
│   └── progress_tracker.py # Quiz history, analytics, recommendations
│
├── data/
│   ├── knowledge_base.csv  # 36 Q&A pairs across 5 topics
│   └── progress.json       # Auto-generated quiz history
│
└── models/
    └── topic_classifier.joblib  # Trained Naive Bayes model (auto-generated)
```

---

## ⚡ Quick Start

### 1. Clone / Download the project

```bash
git clone <repo-url>
cd LearnAI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```bash
python -c "import nltk; nltk.download('all')"
```

### 5. (Optional) Pre-train the ML model

```bash
python train_model.py
```

> The app auto-trains on first launch if the model file is missing.

### 6. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📚 Knowledge Base

The file `data/knowledge_base.csv` contains 36 curated Q&A pairs across five topics:

- **Python** — Variables, Data Types, Functions, OOP
- **Machine Learning** — Supervised/Unsupervised Learning, Neural Networks, Evaluation
- **NLP** — Basics, Preprocessing, Features (TF-IDF, BoW)
- **Data Science** — Statistics, Pandas, NumPy
- **Algorithms** — Sorting, Searching, Complexity

You can extend it by adding rows in the same CSV format.

---

## 🤖 How the AI Works

### Chatbot
1. User inputs a question
2. **Intent detection** — greetings/farewells handled first
3. **Topic classification** — Naive Bayes (trained on KB questions)
4. **Answer retrieval** — TF-IDF cosine similarity against all KB questions
5. Returns the best-matching answer with confidence score

### Quiz Generator
- **Knowledge Base mode** — random Q&A sampling with shuffled options
- **From Notes mode** — heuristic MCQ via POS-tagged noun blanking
- **Mixed mode** — combines both

### Summariser
- Scores each sentence by sum of TF-IDF term weights
- Returns top-N sentences in original document order as bullet points

### Progress Tracker
- Stores results in `data/progress.json`
- Computes per-topic averages, trends, and revision recommendations

---

## 🧪 No Paid APIs

This project uses **zero external AI APIs**. Everything runs offline using open-source Python libraries.

---

## 📝 Extending the Project

- Add more rows to `knowledge_base.csv` to expand chatbot knowledge
- Replace `MultinomialNB` with `SVM` or `RandomForest` in `nlp_engine.py`
- Swap extractive summarisation for a transformer model (e.g. `sumy` or `transformers`)
- Add spaCy NER to extract named entities from uploaded notes

---

## 👩‍💻 Author
Varsha Singh

Built as a beginner-friendly ML + NLP portfolio project.  
Tech: Python · NLTK · scikit-learn · Streamlit · pandas · matplotlib
