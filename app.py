"""
app.py  —  LearnAI: Personal Learning Assistant
Streamlit frontend that ties together all backend modules.

Launch with:  streamlit run app.py
"""

import os
import sys
import io
import random
import textwrap
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── LearnAI backend ──────────────────────────────────────────────────────────
from src.nlp_engine import TextPreprocessor, TFIDFEngine, TopicClassifier
from src.summarizer import Summarizer
from src.quiz_generator import QuizGenerator
from src.progress_tracker import ProgressTracker
from src.chatbot import StudyChatbot

# ── PDF parsing ──────────────────────────────────────────────────────────────
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="LearnAI – Personal Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
/* ── global ── */
html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }

/* ── sidebar ── */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .sidebar-nav-btn {
    background: #1e293b; border-radius: 8px;
    padding: 10px 14px; margin: 4px 0; cursor: pointer;
    transition: background 0.2s;
}

/* ── metric cards ── */
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 18px 22px; text-align: center;
}
.metric-card h2 { margin: 0; font-size: 2rem; color: #3b82f6; }
.metric-card p  { margin: 0; font-size: 0.85rem; color: #64748b; }

/* ── chat bubbles ── */
.chat-user {
    background: #3b82f6; color: white; border-radius: 16px 16px 4px 16px;
    padding: 10px 16px; margin: 6px 0; max-width: 75%; margin-left: auto;
    font-size: 0.95rem;
}
.chat-bot {
    background: #f1f5f9; color: #1e293b; border-radius: 16px 16px 16px 4px;
    padding: 10px 16px; margin: 6px 0; max-width: 80%;
    border-left: 3px solid #3b82f6; font-size: 0.95rem;
}

/* ── quiz option buttons ── */
div[data-testid="stRadio"] label { font-size: 0.95rem; }

/* ── section headers ── */
.section-header {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    color: white; padding: 12px 20px; border-radius: 8px;
    font-size: 1.1rem; font-weight: 600; margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data & model loading  (cached so they load only once)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_knowledge_base() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.csv")
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def get_chatbot(kb: pd.DataFrame) -> StudyChatbot:
    return StudyChatbot(kb)


@st.cache_resource(show_spinner=False)
def get_tracker() -> ProgressTracker:
    return ProgressTracker()


@st.cache_resource(show_spinner=False)
def get_summarizer() -> Summarizer:
    return Summarizer()


@st.cache_resource(show_spinner=False)
def get_quiz_gen() -> QuizGenerator:
    return QuizGenerator()


@st.cache_resource(show_spinner=False)
def get_preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


# =============================================================================
# Session-state initialisation
# =============================================================================
def init_state():
    defaults = {
        "page": "🏠 Home",
        "chat_history": [],
        "uploaded_text": "",
        "quiz_questions": [],
        "quiz_answers": {},
        "quiz_submitted": False,
        "quiz_result": None,
        "quiz_topic": "All",
        "notes_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# =============================================================================
# Helpers
# =============================================================================
def extract_text_from_upload(uploaded_file) -> str:
    """Extract plain text from .txt or .pdf uploads."""
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    if name.endswith(".pdf") and PDF_SUPPORT:
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    return uploaded_file.read().decode("utf-8", errors="replace")


def color_score(pct: float) -> str:
    if pct >= 80:
        return "#22c55e"
    elif pct >= 60:
        return "#f59e0b"
    return "#ef4444"


# =============================================================================
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# =============================================================================
with st.sidebar:
    st.markdown("## 🎓 LearnAI")
    st.markdown("*Personal Learning Assistant*")
    st.divider()

    PAGES = [
        "🏠 Home",
        "💬 AI Chatbot",
        "📄 Upload & Analyse Notes",
        "📝 Quiz",
        "📊 Progress Dashboard",
        "🔍 NLP Explorer",
    ]

    for page in PAGES:
        if st.button(page, use_container_width=True, key=f"nav_{page}"):
            st.session_state.page = page

    st.divider()
    kb = load_knowledge_base()
    st.caption(f"📚 KB: {len(kb)} Q&A pairs")
    st.caption(f"🧠 Topics: {', '.join(kb['topic'].unique())}")

    tracker = get_tracker()
    stats = tracker.overall_stats()
    st.caption(f"✅ Quizzes taken: {stats['quizzes_taken']}")
    if stats['quizzes_taken']:
        st.caption(f"⭐ Avg score: {stats['avg_score']}%")

    st.divider()
    if st.button("🗑️ Reset Progress", use_container_width=True):
        tracker.clear()
        st.success("Progress cleared.")


# =============================================================================
# ── PAGE ROUTING ─────────────────────────────────────────────────────────────
# =============================================================================
page = st.session_state.page

# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="section-header">🎓 Welcome to LearnAI — Your Personal AI Learning Assistant</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    stats = get_tracker().overall_stats()

    with col1:
        st.markdown(f"""<div class="metric-card">
            <h2>{stats['quizzes_taken']}</h2><p>Quizzes Taken</p></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h2>{stats['avg_score']}%</h2><p>Average Score</p></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <h2>{stats['best_score']}%</h2><p>Best Score</p></div>""",
            unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <h2>{stats['total_questions']}</h2><p>Questions Answered</p></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("🚀 Features")
        features = [
            ("💬", "AI Study Chatbot", "Ask any question — get instant answers from the knowledge base using TF-IDF similarity."),
            ("📄", "Upload Notes", "Upload PDF or TXT study notes for NLP analysis and summarisation."),
            ("📝", "Smart Quiz", "Auto-generated quizzes from the KB or your uploaded notes."),
            ("📊", "Progress Dashboard", "Visual analytics of your quiz scores and topic mastery."),
            ("🔍", "NLP Explorer", "See tokenisation, TF-IDF keywords, and text statistics live."),
            ("📌", "Weak Topic Recommendations", "AI identifies which topics need more revision."),
        ]
        for icon, title, desc in features:
            with st.container():
                st.markdown(f"**{icon} {title}** — {desc}")
        
    with col_r:
        st.subheader("📚 Knowledge Base Topics")
        kb_local = load_knowledge_base()
        topic_counts = kb_local["topic"].value_counts()

        fig, ax = plt.subplots(figsize=(4, 4))
        colors = ["#3b82f6", "#8b5cf6", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4"]
        wedges, texts, autotexts = ax.pie(
            topic_counts.values,
            labels=topic_counts.index,
            autopct="%1.0f%%",
            colors=colors[:len(topic_counts)],
            startangle=140,
            textprops={"fontsize": 8},
        )
        ax.set_title("Q&A Distribution", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Weak topics callout
    weak = get_tracker().weak_topics()
    if weak:
        st.warning(f"⚠️ **Topics to revise:** {', '.join(weak)}")
    else:
        st.info("💡 Take a quiz to start tracking your progress!")


# ─────────────────────────────────────────────────────────────────────────────
# AI CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💬 AI Chatbot":
    st.markdown('<div class="section-header">💬 AI Study Chatbot</div>',
                unsafe_allow_html=True)
    st.caption("Ask questions about Python, Machine Learning, NLP, Data Science, or Algorithms.")

    kb_local = load_knowledge_base()
    bot = get_chatbot(kb_local)

    # Add uploaded notes to chatbot knowledge if available
    if st.session_state.uploaded_text:
        bot.add_document(st.session_state.uploaded_text)

    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""<div class="chat-bot">
                👋 Hello! I'm your AI study assistant.<br>
                Ask me anything about <b>Python, ML, NLP, Data Science,</b> or <b>Algorithms</b>!
            </div>""", unsafe_allow_html=True)
        else:
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f'<div class="chat-user">{msg}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot">{msg}</div>',
                                unsafe_allow_html=True)

    st.divider()

    # Input row
    col_inp, col_btn, col_clr = st.columns([6, 1, 1])
    with col_inp:
        user_input = st.text_input("Your question:", label_visibility="collapsed",
                                   placeholder="e.g. What is TF-IDF?",
                                   key="chat_input")
    with col_btn:
        send = st.button("Send ➤", use_container_width=True)
    with col_clr:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_input.strip():
        response = bot.respond(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response["reply"]))
        st.rerun()

    # Suggested questions
    with st.expander("💡 Suggested questions"):
        suggestions = [
            "What is TF-IDF?",
            "Explain overfitting in machine learning.",
            "What is the difference between a list and a tuple?",
            "What is backpropagation?",
            "What are stop words in NLP?",
            "What is Big-O notation?",
            "What is a DataFrame in pandas?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}"):
                response = bot.respond(s)
                st.session_state.chat_history.append(("user", s))
                st.session_state.chat_history.append(("bot", response["reply"]))
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD & ANALYSE NOTES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📄 Upload & Analyse Notes":
    st.markdown('<div class="section-header">📄 Upload & Analyse Study Notes</div>',
                unsafe_allow_html=True)

    summarizer = get_summarizer()
    preprocessor = get_preprocessor()

    tab_upload, tab_paste = st.tabs(["📁 Upload File", "✏️ Paste Text"])

    with tab_upload:
        uploaded = st.file_uploader("Upload PDF or TXT file",
                                    type=["pdf", "txt"],
                                    help="Max 10 MB recommended")
        if uploaded:
            with st.spinner("Extracting text…"):
                text = extract_text_from_upload(uploaded)
            if text.strip():
                st.session_state.notes_text = text
                st.success(f"✅ Extracted {len(text):,} characters from **{uploaded.name}**")
            else:
                st.error("Could not extract text. Try a different file.")

    with tab_paste:
        pasted = st.text_area("Paste your notes here:", height=200,
                              placeholder="Paste any study text…")
        if st.button("Analyse Pasted Text"):
            st.session_state.notes_text = pasted

    text = st.session_state.notes_text
    if not text.strip():
        st.info("Upload or paste notes above to begin analysis.")
        st.stop()

    st.markdown("---")

    # ── Text stats ──────────────────────────────────────────────────────────
    st.subheader("📊 Text Statistics")
    stats = preprocessor.get_token_stats(text)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Characters", f"{len(text):,}")
    c2.metric("Sentences", stats["sentences"])
    c3.metric("Total Words", stats["total_tokens"])
    c4.metric("Unique Words", stats["unique_tokens"])
    c5.metric("Avg Word Len", f"{stats['avg_word_length']:.1f}")

    # ── Summary ─────────────────────────────────────────────────────────────
    st.subheader("📝 Automatic Summary")
    ratio = st.slider("Summary ratio (% sentences to keep)", 10, 60, 30, 5) / 100
    with st.spinner("Summarising…"):
        summary = summarizer.summarize(text, ratio=ratio)
    st.markdown(summary)

    # ── Key points ──────────────────────────────────────────────────────────
    st.subheader("🔑 Key Points")
    key_pts = summarizer.key_points(text, top_k=5)
    for i, pt in enumerate(key_pts, 1):
        st.markdown(f"**{i}.** {pt}")

    # ── TF-IDF Keywords ─────────────────────────────────────────────────────
    st.subheader("🏷️ Top Keywords (TF-IDF)")
    keywords = summarizer.extract_topics_from_text(text, top_k=15)
    if keywords:
        kw_df = pd.DataFrame({"Keyword": keywords,
                               "Rank": range(1, len(keywords) + 1)})
        col_kw, col_bar = st.columns([1, 2])
        col_kw.dataframe(kw_df, use_container_width=True, hide_index=True)

        freq = summarizer.word_frequency(text)
        top_words = dict(list(freq.items())[:12])
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.barh(list(top_words.keys())[::-1], list(top_words.values())[::-1],
                 color="#3b82f6")
        ax2.set_xlabel("Frequency")
        ax2.set_title("Word Frequency", fontsize=10)
        plt.tight_layout()
        col_bar.pyplot(fig2)
        plt.close()

    # ── Add to chatbot ───────────────────────────────────────────────────────
    if st.button("➕ Add notes to Chatbot knowledge base"):
        st.session_state.uploaded_text = text
        st.success("Notes added! The chatbot can now answer questions from your notes.")


# ─────────────────────────────────────────────────────────────────────────────
# QUIZ
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📝 Quiz":
    st.markdown('<div class="section-header">📝 Smart Quiz Generator</div>',
                unsafe_allow_html=True)

    kb_local = load_knowledge_base()
    quiz_gen = get_quiz_gen()
    tracker = get_tracker()

    topics = ["All"] + sorted(kb_local["topic"].unique().tolist())

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        selected_topic = st.selectbox("Topic", topics, key="quiz_topic_sel")
    with col_s2:
        n_questions = st.slider("Number of questions", 3, 10, 5)
    with col_s3:
        quiz_mode = st.radio("Mode", ["Knowledge Base", "From Notes", "Mixed"],
                             horizontal=True)

    generate = st.button("🎲 Generate New Quiz", type="primary")

    if generate:
        notes = st.session_state.notes_text
        with st.spinner("Generating quiz…"):
            if quiz_mode == "Knowledge Base":
                qs = quiz_gen.from_knowledge_base(kb_local, topic=selected_topic,
                                                  n=n_questions)
            elif quiz_mode == "From Notes" and notes.strip():
                qs = quiz_gen.from_text(notes, n=n_questions)
            elif quiz_mode == "Mixed" and notes.strip():
                qs = quiz_gen.generate_mixed_quiz(kb_local, notes,
                                                  topic=selected_topic,
                                                  n=n_questions)
            else:
                qs = quiz_gen.from_knowledge_base(kb_local, topic=selected_topic,
                                                  n=n_questions)

        if qs:
            st.session_state.quiz_questions = qs
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_result = None
        else:
            st.warning("No questions generated. Try a different topic or upload notes.")

    qs = st.session_state.quiz_questions

    if not qs:
        st.info("Click **Generate New Quiz** to start.")
    else:
        st.markdown(f"### 📋 Quiz — {len(qs)} Question{'s' if len(qs) > 1 else ''}")

        if not st.session_state.quiz_submitted:
            with st.form("quiz_form"):
                answers = {}
                for i, q in enumerate(qs):
                    st.markdown(f"**Q{i+1}. {q['question']}**")
                    st.caption(f"Topic: {q['topic']}")
                    if q["type"] == "mcq" and q["options"]:
                        answers[i] = st.radio(
                            f"Select answer for Q{i+1}",
                            q["options"],
                            key=f"q_{i}",
                            label_visibility="collapsed",
                        )
                    else:
                        answers[i] = st.text_area(
                            f"Your answer for Q{i+1}",
                            key=f"qa_{i}",
                            height=80,
                            label_visibility="collapsed",
                            placeholder="Type your answer here…",
                        )
                    st.divider()

                submitted = st.form_submit_button("✅ Submit Quiz", type="primary")

            if submitted:
                user_ans = [answers.get(i, "") for i in range(len(qs))]
                result = quiz_gen.score_attempt(qs, user_ans)
                st.session_state.quiz_submitted = True
                st.session_state.quiz_result = result
                topic_label = selected_topic if selected_topic != "All" else "Mixed"
                tracker.record_quiz(
                    topic=topic_label,
                    score=result["score"],
                    total=result["total"],
                    breakdown=result["breakdown"],
                )
                st.rerun()

        else:
            result = st.session_state.quiz_result
            pct = result["percentage"]
            clr = color_score(pct)

            # Score banner
            st.markdown(
                f"<div style='background:{clr};color:white;padding:16px;border-radius:10px;"
                f"text-align:center;font-size:1.6rem;font-weight:700;margin-bottom:16px'>"
                f"Score: {result['score']} / {result['total']} ({pct}%)</div>",
                unsafe_allow_html=True,
            )

            if pct >= 80:
                st.success("🌟 Excellent work! You've mastered this topic.")
            elif pct >= 60:
                st.warning("👍 Good effort! A few more revisions and you'll nail it.")
            else:
                st.error("📖 Keep studying! Review the answers below.")

            # Breakdown
            st.markdown("### 📋 Answer Review")
            for i, bd in enumerate(result["breakdown"]):
                icon = "✅" if bd["is_correct"] else "❌"
                with st.expander(f"{icon} Q{i+1}: {bd['question'][:80]}…"):
                    st.markdown(f"**Your answer:** {bd['your_answer'] or '*(no answer)*'}")
                    st.markdown(f"**Correct answer:** {bd['correct_answer']}")
                    st.markdown(f"*Topic: {bd['topic']}*")

            if st.button("🔄 Try Another Quiz"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_submitted = False
                st.session_state.quiz_result = None
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Progress Dashboard":
    st.markdown('<div class="section-header">📊 Learning Progress Dashboard</div>',
                unsafe_allow_html=True)

    tracker = get_tracker()
    stats = tracker.overall_stats()
    df_scores = tracker.score_over_time()
    topic_stats = tracker.topic_stats()

    if stats["quizzes_taken"] == 0:
        st.info("🎯 No quiz data yet. Take a quiz to see your progress here!")
        st.stop()

    # ── Top metrics ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quizzes Taken", stats["quizzes_taken"])
    c2.metric("Avg Score", f"{stats['avg_score']}%")
    c3.metric("Best Score", f"{stats['best_score']}%")
    c4.metric("Total Questions", stats["total_questions"])

    st.markdown("---")
    col_left, col_right = st.columns(2)

    # ── Score over time ──────────────────────────────────────────────────────
    with col_left:
        st.subheader("📈 Score Over Time")
        if len(df_scores) >= 2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(df_scores["timestamp"], df_scores["percentage"],
                    marker="o", color="#3b82f6", linewidth=2, markersize=6)
            ax.axhline(60, color="#f59e0b", linestyle="--", alpha=0.6, label="Pass (60%)")
            ax.axhline(80, color="#22c55e", linestyle="--", alpha=0.6, label="Good (80%)")
            ax.set_ylim(0, 105)
            ax.set_ylabel("Score (%)")
            ax.set_title("Quiz Scores")
            ax.legend(fontsize=8)
            plt.xticks(rotation=30, fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Take at least 2 quizzes to see a trend line.")

    # ── Per-topic bar chart ──────────────────────────────────────────────────
    with col_right:
        st.subheader("🎯 Average Score by Topic")
        if not topic_stats.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            colors = [color_score(s) for s in topic_stats["avg_score"]]
            bars = ax2.bar(topic_stats["topic"], topic_stats["avg_score"],
                           color=colors, edgecolor="white", linewidth=0.8)
            ax2.set_ylim(0, 110)
            ax2.set_ylabel("Avg Score (%)")
            ax2.set_title("Topic Mastery")
            for bar, val in zip(bars, topic_stats["avg_score"]):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 2, f"{val:.0f}%",
                         ha="center", va="bottom", fontsize=8)
            plt.xticks(rotation=25, ha="right", fontsize=8)
            patches = [
                mpatches.Patch(color="#22c55e", label="Strong (≥80%)"),
                mpatches.Patch(color="#f59e0b", label="Average (60-80%)"),
                mpatches.Patch(color="#ef4444", label="Weak (<60%)"),
            ]
            ax2.legend(handles=patches, fontsize=7)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    # ── Topic stats table ────────────────────────────────────────────────────
    st.subheader("📋 Topic Statistics")
    if not topic_stats.empty:
        display_df = topic_stats[["topic", "avg_score", "attempts", "trend"]].copy()
        display_df.columns = ["Topic", "Avg Score (%)", "Attempts", "Trend (pp)"]
        st.dataframe(display_df.set_index("Topic"), use_container_width=True)

    # ── Recommendations ──────────────────────────────────────────────────────
    st.markdown("---")
    col_w, col_s = st.columns(2)

    with col_w:
        st.subheader("⚠️ Topics to Revise")
        weak = tracker.weak_topics()
        if weak:
            for t in weak:
                st.markdown(f"🔴 **{t}** — score below 60%")
        else:
            st.success("✅ All topics are above the threshold — great work!")

    with col_s:
        st.subheader("🏆 Strong Topics")
        strong = tracker.strong_topics()
        if strong:
            for t in strong:
                st.markdown(f"🟢 **{t}** — keep it up!")
        else:
            st.info("Keep practising to build strong topic mastery.")

    # ── Recommendations ranking ──────────────────────────────────────────────
    recs = tracker.revision_recommendations()
    if recs:
        st.subheader("📌 Revision Priority List")
        for i, r in enumerate(recs, 1):
            trend_arrow = "↑" if r["trend"] > 0 else ("↓" if r["trend"] < 0 else "→")
            trend_color = "#22c55e" if r["trend"] > 0 else ("#ef4444" if r["trend"] < 0 else "#64748b")
            st.markdown(
                f"**{i}. {r['topic']}** — "
                f"Avg: `{r['avg_score']:.1f}%` | "
                f"Attempts: `{r['attempts']}` | "
                f"Trend: <span style='color:{trend_color}'>{trend_arrow} {abs(r['trend']):.1f}pp</span>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# NLP EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 NLP Explorer":
    st.markdown('<div class="section-header">🔍 NLP Explorer — See the AI at Work</div>',
                unsafe_allow_html=True)
    st.caption("Type any text to see how the NLP pipeline processes it step by step.")

    preprocessor = get_preprocessor()
    summarizer = get_summarizer()
    kb_local = load_knowledge_base()

    input_text = st.text_area("Enter text to analyse:",
                              value="Machine learning is a branch of artificial intelligence "
                                    "that uses algorithms to learn patterns from data.",
                              height=120)

    if not input_text.strip():
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔤 Tokenisation",
        "🧹 Preprocessing",
        "📊 TF-IDF",
        "🏷️ Topic Detection",
        "📐 Text Similarity",
    ])

    # ── Tokenisation ────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Tokenisation & Statistics")
        stats = preprocessor.get_token_stats(input_text)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tokens", stats["total_tokens"])
        c2.metric("Unique Tokens", stats["unique_tokens"])
        c3.metric("Sentences", stats["sentences"])
        c4.metric("Avg Word Length", f"{stats['avg_word_length']:.2f}")

        tokens = preprocessor.tokenize(input_text)
        st.markdown("**Raw tokens:**")
        st.code(" | ".join(tokens[:60]))

        no_stop = preprocessor.remove_stopwords(tokens)
        st.markdown("**After stop-word removal:**")
        st.code(" | ".join(no_stop[:60]))

    # ── Preprocessing ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Text Cleaning & Normalisation")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Lemmatised output:**")
            tokens2 = preprocessor.tokenize(input_text)
            tokens2 = preprocessor.remove_stopwords(tokens2)
            lemmas = preprocessor.lemmatize(tokens2)
            st.info(" ".join(lemmas))

        with col_b:
            st.markdown("**Stemmed output:**")
            stems = preprocessor.stem(tokens2)
            st.info(" ".join(stems))

        st.markdown("**Full preprocessed string (ready for ML):**")
        st.success(preprocessor.preprocess(input_text))

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("TF-IDF Keyword Extraction")
        keywords = summarizer.extract_topics_from_text(input_text, top_k=12)
        freq = summarizer.word_frequency(input_text)

        if keywords:
            col_kw, col_chart = st.columns([1, 2])
            col_kw.markdown("**Top keywords:**")
            for i, kw in enumerate(keywords, 1):
                col_kw.markdown(f"`{i}.` **{kw}**")

            if freq:
                top_f = dict(list(freq.items())[:10])
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(list(top_f.keys())[::-1], list(top_f.values())[::-1],
                        color="#8b5cf6")
                ax.set_title("Word Frequency", fontsize=10)
                plt.tight_layout()
                col_chart.pyplot(fig)
                plt.close()
        else:
            st.warning("Text too short for meaningful TF-IDF extraction.")

    # ── Topic Detection ───────────────────────────────────────────────────────
    with tab4:
        st.subheader("Topic / Intent Classification")
        clf = TopicClassifier()
        if not clf.load():
            clf.train(kb_local)
        predicted = clf.predict(input_text)
        probs = clf.predict_proba(input_text)

        st.markdown(f"**Predicted topic:** 🏷️ `{predicted}`")

        if probs:
            prob_df = (
                pd.DataFrame(list(probs.items()), columns=["Topic", "Probability"])
                .sort_values("Probability", ascending=False)
            )
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            ax3.barh(prob_df["Topic"][::-1], prob_df["Probability"][::-1],
                     color="#3b82f6")
            ax3.set_xlabel("Probability")
            ax3.set_title("Topic Probability Distribution", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

    # ── Text Similarity ──────────────────────────────────────────────────────
    with tab5:
        st.subheader("TF-IDF Cosine Similarity")
        st.caption("Compare your text against the knowledge base or another custom text.")

        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            compare_text = st.text_area("Compare with:",
                                        value="Deep learning uses neural networks to process data.",
                                        height=100, key="sim_compare")
        with col_sim2:
            st.markdown("**Similarity score:**")
            tfidf_eng = TFIDFEngine()
            tfidf_eng.fit([input_text, compare_text])
            sim = tfidf_eng.similarity(input_text, compare_text)
            bar_color = "#22c55e" if sim > 0.5 else ("#f59e0b" if sim > 0.2 else "#ef4444")
            st.markdown(
                f"<div style='background:{bar_color};color:white;border-radius:8px;"
                f"padding:14px;text-align:center;font-size:1.8rem;font-weight:700'>"
                f"{sim:.3f}</div>",
                unsafe_allow_html=True,
            )
            st.caption("0 = completely different · 1 = identical")

        st.markdown("---")
        st.markdown("**Most similar KB questions to your text:**")
        engine = TFIDFEngine()
        engine.fit(kb_local["question"].tolist())
        results = engine.search(input_text, top_k=5)
        for rank, (idx, score) in enumerate(results, 1):
            row = kb_local.iloc[idx]
            st.markdown(
                f"**{rank}.** [{score:.3f}] *{row['question']}* "
                f"→ `{row['topic']}`"
            )
