import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import os
from summarize_mmr import summarize_mmr

# Download NLTK punkt if not already
nltk.download('punkt')

# --- CONFIG ---
st.set_page_config(page_title="News Summarization App", layout="wide")

# --- DATA LOADING ---
@st.cache_data

def load_data():
    df = pd.read_csv("dataset_artikel_final.csv", index_col=0)
    df['Text'] = df['Text'].str.strip('\'"')
    df = df.dropna()
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Dataset Overview", "EDA & Visualizations", "Summarization Demo"])

# --- DATASET OVERVIEW ---
if section == "Dataset Overview":
    st.title("📰 News Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Columns:**", ', '.join(df.columns))
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")
    st.write("**Column Descriptions:**")
    st.markdown("""
    - **Title**: Judul Artikel
    - **Text**: Isi lengkap dari artikel
    - **Summary**: Ringkasan artikel yang dibuat oleh penulis
    - **Link News**: Source Link Menuju artikel asli
    """)

# --- EDA & VISUALIZATIONS ---
elif section == "EDA & Visualizations":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.subheader("Distribusi panjang artikel")
    
    df['char_count'] = df['Text'].astype(str).apply(len)
    df['word_count'] = df['Text'].astype(str).apply(lambda x: len(x.split()))
    df['sent_count'] = df['Text'].astype(str).apply(lambda x: len(nltk.sent_tokenize(x)))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Karakter per artikel**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['char_count'], bins=30, color='#4F8BF9')
        st.pyplot(fig)
        st.write(f"Mean: {df['char_count'].mean():.1f}")
    with col2:
        st.write("**Kata per artikel**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['word_count'], bins=30, color='#F9A34F')
        st.pyplot(fig)
        st.write(f"Mean: {df['word_count'].mean():.1f}")
    with col3:
        st.write("**Kalimat per artikel**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['sent_count'], bins=30, color='#4FF9A3')
        st.pyplot(fig)
        st.write(f"Mean: {df['sent_count'].mean():.1f}")

    st.markdown("---")
    st.subheader("Word Clouds")
    ngram = st.radio("WordCloud Type", ["Unigram", "Bigram"])
    text = ' '.join(df['Text'].astype(str))
    if ngram == "Unigram":
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english')
        X = vectorizer.fit_transform(df['Text'].astype(str))
        bigrams = vectorizer.get_feature_names_out()
        freqs = X.sum(axis=0).A1
        bigram_freq = dict(zip(bigrams, freqs))
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigram_freq)
    st.image(wc.to_array(), use_container_width=True)

# --- SUMMARIZATION DEMO ---
else:
    st.title("📝 MMR-Based Summarization")
    st.write("Pilihlah artikel yang tersedia dari dataset atau paste artikel yang ingin di ringkas")
    tab1, tab2 = st.tabs(["Pilih Dari Dataset", "Paste Custom Text"])

    with tab1:
        if 'Title' not in df.columns:
            st.error("Kolom 'Title' tidak ditemukan di dataset.")
        else:
            idx = st.selectbox(
                "Pilih artikel:",
                range(len(df)),
                format_func=lambda i: df.iloc[i]['Title'] if pd.notnull(df.iloc[i]['Title']) else f"Artikel {i}"
            )
            article = df.iloc[idx]['Text']
            title = df.iloc[idx]['Title'] if 'Title' in df.columns else None
            st.write("**Original Article:**")
            st.info(article)
            n_sent = st.slider("Jumlah kalimat dalam ringkasan", 1, 7, 3)
            lambda_param = st.slider("MMR Lambda (Seberapa Relevan dengan query)", 0.0, 1.0, 0.5, 0.05)
            mode = st.radio("Mode Summarization", ["Default", "Judul sebagai Query"], horizontal=True)
            st.caption("Mode 'Judul sebagai Query' akan menggunakan judul artikel sebagai query untuk ringkasan yang lebih fokus pada topik judul.")
            if st.button("Summarize", key="summarize1"):
                with st.spinner("Summarizing..."):
                    try:
                        if mode == "Judul sebagai Query" and title:
                            summary = summarize_mmr(article, summary_length=n_sent, lambda_param=lambda_param, query=title)
                        else:
                            summary = summarize_mmr(article, summary_length=n_sent, lambda_param=lambda_param)
                        st.success("\n".join(summary))
                        if 'Summary' in df.columns:
                            st.write("**Reference Summary:**")
                            st.info(df.iloc[idx]['Summary'])
                    except Exception as e:
                        st.error(f"Error: {e}")
    with tab2:
        user_text = st.text_area("Paste artikel yang ingin di ringkas:")
        custom_query = st.text_input("Opsional: Masukkan judul atau query untuk ringkasan (biarkan kosong jika tidak ingin)")
        n_sent2 = st.slider("Jumlah kalimat dalam ringkasan", 1, 7, 3, key="n_sent2")
        lambda_param2 = st.slider("MMR Lambda (Seberapa Relevan dengan query)", 0.0, 1.0, 0.5, 0.05, key="lambda2")
        st.caption("Jika Anda mengisi judul/query, ringkasan akan lebih fokus pada topik tersebut.")
        if st.button("Summarize", key="summarize2"):
            if not user_text.strip():
                st.warning("Please paste some text.")
            else:
                with st.spinner("Summarizing..."):
                    try:
                        if custom_query.strip():
                            summary = summarize_mmr(user_text, summary_length=n_sent2, lambda_param=lambda_param2, query=custom_query)
                        else:
                            summary = summarize_mmr(user_text, summary_length=n_sent2, lambda_param=lambda_param2)
                        st.success("\n".join(summary))
                    except Exception as e:
                        st.error(f"Error: {e}")

st.markdown("""
---
### Made By ❤️ : 
**Ivan Roisus Salam (230411200206)** \n
**Hasanatun Fajariya (220411100064)** \n
**Putri Qurratu Aini (220411100126)** \n
""")
