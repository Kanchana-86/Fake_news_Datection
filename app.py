import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# --- Page Config ---
st.set_page_config(
    page_title="Veritas AI | Fake News Forensics",
    page_icon="⚖️",
    layout="wide"
)

# --- Custom CSS for Beautiful Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap');
   
    * {
        font-family: 'Inter', sans-serif;
    }
   
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
   
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
   
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
   
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #2d3748;
    }
   
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        height: 3.5em;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }
   
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
   
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
   
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
   
    .metric-card h3 {
        font-size: 0.9rem;
        font-weight: 600;
        color: #718096;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
   
    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        font-family: 'Poppins', sans-serif;
    }
   
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
        padding: 15px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
   
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
   
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
   
    .info-box h4 {
        color: white !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        margin-bottom: 15px;
    }
   
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f7fafc;
        border-radius: 12px;
        padding: 8px;
    }
   
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        padding: 12px 24px;
        background-color: transparent;
    }
   
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
   
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
   
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 2rem 0;
    }
   
    .subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        color: #4a5568;
        font-weight: 400;
        margin-bottom: 2rem;
    }
   
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Models (Cached for performance) ---
@st.cache_resource
def load_models():
    try:
        with open('model2.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidfvect2.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, vectorizer = load_models()

# --- Helper Functions ---
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def explain_prediction(vectorizer, model, text_input):
    try:
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_.flatten()
        tfidf_matrix = vectorizer.transform([text_input])
        feature_index = tfidf_matrix.indices
       
        word_impacts = []
        for idx in feature_index:
            word = feature_names[idx]
            score = coefs[idx] * tfidf_matrix[0, idx]
            word_impacts.append({'Word': word, 'Impact': score})
       
        df_impact = pd.DataFrame(word_impacts)
        if not df_impact.empty:
            df_impact = df_impact.sort_values(by='Impact', ascending=False)
        return df_impact
    except Exception as e:
        st.warning(f"Could not generate explanation: {e}")
        return pd.DataFrame()

# --- Main Interface ---
st.markdown("<h1>⚖️ Veritas AI: Fake News Forensics</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced NLP Analysis Dashboard for News Credibility</div>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h3>📰 News Analysis</h3>", unsafe_allow_html=True)
    news_text = st.text_area("Paste the news article text here:", height=250, placeholder="Enter full article text for forensic analysis...")
   
    analyze_btn = st.button("🔍 Analyze Credibility")

with col2:
    st.markdown("""
    <div class='info-box'>
        <h4>ℹ️ How It Works</h4>
        <p style='margin-bottom: 10px;'><strong>1. Veracity Check</strong><br/>PassiveAggressive Classifier trained on 20k+ articles</p>
        <p style='margin-bottom: 10px;'><strong>2. Linguistic Forensics</strong><br/>Analyzes sentence structure for sensationalism</p>
        <p style='margin-bottom: 0;'><strong>3. Bias Meter</strong><br/>Detects subjective or emotional language</p>
    </div>
    """, unsafe_allow_html=True)

if analyze_btn and news_text:
    if model and vectorizer:
        with st.spinner("🔬 Running Forensics Pipeline..."):
            # 1. Preprocessing & Prediction
            processed_text = wordopt(news_text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction_idx = model.predict(vectorized_text)[0]
           
            # Handle Decision Function for Confidence
            try:
                decision_val = model.decision_function(vectorized_text)[0]
                confidence = 1 / (1 + np.exp(-decision_val))
               
                if prediction_idx == 0:
                    confidence_score = (1 - confidence) * 100
                    result_label = "FAKE NEWS DETECTED"
                    result_color = "#ef4444"
                    border_gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
                else:
                    confidence_score = confidence * 100
                    result_label = "LIKELY REAL NEWS"
                    result_color = "#10b981"
                    border_gradient = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
                   
            except:
                result_label = "FAKE" if prediction_idx == 0 else "REAL"
                confidence_score = 0
                result_color = "#6b7280"
                border_gradient = "linear-gradient(135deg, #6b7280 0%, #4b5563 100%)"

            # 2. Sentiment Analysis
            polarity, subjectivity = get_sentiment(news_text)

            # --- DISPLAY RESULTS ---
           
            # Top Metric Cards
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class='metric-card' style='border-top: 4px solid {result_color};'>
                    <h3>Verdict</h3>
                    <h2 style='color:{result_color};'>{result_label}</h2>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='metric-card' style='border-top: 4px solid #3b82f6;'>
                    <h3>Confidence</h3>
                    <h2 style='color:#3b82f6;'>{confidence_score:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                subj_color = "#f59e0b" if subjectivity > 0.5 else "#10b981"
                st.markdown(f"""
                <div class='metric-card' style='border-top: 4px solid {subj_color};'>
                    <h3>Subjectivity</h3>
                    <h2 style='color:{subj_color};'>{subjectivity:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # --- VISUALIZATION TABS ---
            tab1, tab2, tab3 = st.tabs(["🔍 Explainability", "📊 Bias Analysis", "☁️ Word Cloud"])

            with tab1:
                st.markdown("<h3>What triggered the model?</h3>", unsafe_allow_html=True)
                explanation_df = explain_prediction(vectorizer, model, processed_text)
               
                if not explanation_df.empty:
                    col_l, col_r = st.columns(2)
                   
                    with col_l:
                        st.markdown("#### 🚫 Words signaling 'Fake'")
                        fake_drivers = explanation_df[explanation_df['Impact'] < 0].sort_values(by='Impact').head(10)
                        fake_drivers['AbsImpact'] = fake_drivers['Impact'].abs()
                        if not fake_drivers.empty:
                            fig1, ax1 = plt.subplots(figsize=(5, 4))
                            sns.barplot(x='AbsImpact', y='Word', data=fake_drivers, palette="Reds_r", ax=ax1)
                            ax1.set_xlabel("Impact Score", fontsize=12, fontweight='bold')
                            ax1.set_ylabel("")
                            plt.tight_layout()
                            st.pyplot(fig1)
                        else:
                            st.info("No strong indicators for Fake found.")

                    with col_r:
                        st.markdown("#### ✅ Words signaling 'Real'")
                        real_drivers = explanation_df[explanation_df['Impact'] > 0].head(10)
                        if not real_drivers.empty:
                            fig2, ax2 = plt.subplots(figsize=(5, 4))
                            sns.barplot(x='Impact', y='Word', data=real_drivers, palette="Greens_r", ax=ax2)
                            ax2.set_xlabel("Impact Score", fontsize=12, fontweight='bold')
                            ax2.set_ylabel("")
                            plt.tight_layout()
                            st.pyplot(fig2)
                        else:
                            st.info("No strong indicators for Real found.")
                else:
                    st.write("Could not extract feature importance.")

            with tab2:
                st.markdown("<h3>Bias & Tone Analysis</h3>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**😠 ← Sentiment Polarity → 😊**")
                    st.progress(min(max((polarity + 1) / 2, 0.0), 1.0))
                    st.caption("Negative/Hate Speech ← → Positive/Promotional")
               
                with c2:
                    st.markdown("**📊 Subjectivity Score**")
                    st.progress(subjectivity)
                    st.caption("Objective/Factual ← → Opinionated/Emotional")

            with tab3:
                st.markdown("<h3>Key Topics & Terms</h3>", unsafe_allow_html=True)
                wc = WordCloud(width=800, height=400, background_color='white',
                               colormap='viridis', relative_scaling=0.5).generate(processed_text)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis("off")
                plt.tight_layout()
                st.pyplot(fig_wc)

    else:
        st.error("❌ Models not loaded. Please check .pkl files.")
