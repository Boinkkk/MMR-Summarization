import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re
from wordcloud import WordCloud
import nltk

def tokenize_sentence(text):
    return nltk.sent_tokenize(text)

def compute_tf(tokenized_sentences, all_words):
    tf_matrix = []
    for sentence in tokenized_sentences:
        tf_row = [sentence.count(word) for word in all_words]
        tf_matrix.append(tf_row)
    return np.array(tf_matrix)

def compute_idf(tokenized_sentences, all_words):
    N = len(tokenized_sentences)
    idf_values = []
    for word in all_words:
        # jumlah kalimat yang mengandung kata tersebut
        df = sum(1 for sentence in tokenized_sentences if word in sentence)
        # rumus smoothing IDF
        idf = math.log((N + 1) / (df + 1)) + 1
        idf_values.append(idf)
    return np.array(idf_values)

def compute_tfidf(tf_matrix, idf_values):
    tfidf_matrix = []
    for tf_row in tf_matrix:
        tfidf_row = [tf * idf for tf, idf in zip(tf_row, idf_values)]
        tfidf_matrix.append(tfidf_row)
    return np.array(tfidf_matrix)

def dot(vektor1, vektor2):
    if len(vektor1) != len(vektor2):
        raise ValueError("Vektor untuk dot product harus memiliki panjang yang sama.")
    return np.sum(vektor1 * vektor2)

def magnitudo(vektor):
    return np.sqrt(np.sum(vektor**2))
  
def cosine_similarity(vektor1, vektor2):
    dot_product = dot(vektor1, vektor2)
    norm_vektor1 = magnitudo(vektor1)
    norm_vektor2 = magnitudo(vektor2)

    if norm_vektor1 == 0 or norm_vektor2 == 0:
        return 0.0
    else:
        similarity = dot_product / (norm_vektor1 * norm_vektor2)
        return similarity

def cosine_similarity_matrix(matriks_vektor):
    jumlah_vektor = matriks_vektor.shape[0]
    similarity_matrix = np.zeros((jumlah_vektor, jumlah_vektor))

    for i in range(jumlah_vektor):
        for j in range(jumlah_vektor):
            vektor_i = matriks_vektor[i]
            vektor_j = matriks_vektor[j]
            similarity_matrix[i, j] = cosine_similarity(vektor_i, vektor_j)

    return similarity_matrix

def mmr(doc_embedding, sentence_embeddings, sentences, top_n=3, lambda_param=0.5):
    """
    Algoritma MMR yang diperbaiki untuk menggunakan cosine_similarity yang benar
    """
    selected = []
    candidates = list(range(len(sentences)))

    # Step 1: Pilih kalimat paling relevan terhadap dokumen
    doc_sim = []
    for i in range(len(sentence_embeddings)):
        sim = cosine_similarity(doc_embedding, sentence_embeddings[i])
        doc_sim.append(sim)
    doc_sim = np.array(doc_sim)
    
    selected.append(np.argmax(doc_sim))
    candidates.remove(selected[0])

    for _ in range(top_n - 1):
        mmr_scores = []
        for candidate in candidates:
            sim_to_doc = doc_sim[candidate]
            
            # Hitung similarity dengan semua kalimat yang sudah dipilih
            similarities_to_selected = []
            for i in selected:
                sim = cosine_similarity(sentence_embeddings[candidate], sentence_embeddings[i])
                similarities_to_selected.append(sim)
            
            sim_to_selected = max(similarities_to_selected) if similarities_to_selected else 0
            mmr_score = lambda_param * sim_to_doc - (1 - lambda_param) * sim_to_selected
            mmr_scores.append((candidate, mmr_score))
        
        # Pilih dengan skor MMR tertinggi
        best_candidate = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_candidate)
        candidates.remove(best_candidate)

    return [sentences[i] for i in selected]

def summarize_mmr(text, summary_length=3, lambda_param=0.5, query=None):
    """
    Fungsi untuk membuat ringkasan teks menggunakan algoritma Maximal Marginal Relevance (MMR)
    menggunakan hanya numpy dan fungsi-fungsi yang sudah ada.
    
    Parameters:
    - text: string, teks yang akan diringkas
    - summary_length: int, jumlah kalimat dalam ringkasan (default=3)
    - lambda_param: float, parameter lambda untuk MMR (default=0.5)
                   nilai tinggi = fokus pada relevance, nilai rendah = fokus pada diversity
    - query: string, opsional. Jika diberikan, digunakan sebagai query (misal judul artikel) untuk MMR
    
    Returns:
    - list: kalimat-kalimat terpilih untuk ringkasan
    """
    # Tokenisasi kalimat
    sentences = np.array(tokenize_sentence(text))
    # Tokenisasi kata untuk setiap kalimat
    tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    # Buat vocabulary
    all_words = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    # Hitung TF matrix menggunakan fungsi yang sudah ada
    tf_matrix = compute_tf(tokenized_sentences, all_words)
    # Hitung IDF values menggunakan fungsi yang sudah ada
    idf_values = compute_idf(tokenized_sentences, all_words)
    # Hitung TF-IDF matrix menggunakan fungsi yang sudah ada
    tfidf_matrix = compute_tfidf(tf_matrix, idf_values)
    # Vektor representasi dokumen
    if query is not None and isinstance(query, str) and query.strip():
        # Tokenisasi dan vektorisasi query
        query_tokens = nltk.word_tokenize(query.lower())
        query_tf = np.array([query_tokens.count(word) for word in all_words])
        query_tfidf = query_tf * idf_values
        doc_embedding = query_tfidf
    else:
        doc_embedding = tfidf_matrix.mean(axis=0)
    # Panggil fungsi MMR dengan parameter yang benar
    return mmr(doc_embedding, tfidf_matrix, sentences, summary_length, lambda_param)


if __name__ == "__main__":
    # Test fungsi summarize_mmr
    text = """
    Indonesia adalah negara kepulauan terbesar di dunia yang terletak di Asia Tenggara. 
    Negara ini memiliki lebih dari 17.000 pulau dan merupakan rumah bagi lebih dari 270 juta orang. 
    Ibu kotanya adalah Jakarta, kota metropolitan yang padat. 
    Bahasa resmi negara ini adalah Bahasa Indonesia. 
    Indonesia juga dikenal dengan keanekaragaman budaya dan sumber daya alamnya yang melimpah.
    """
    
    print("=== TEST SUMMARIZE MMR ===")
    try:
        summary = summarize_mmr(text, summary_length=3, lambda_param=0.7)
        print("\n=== RINGKASAN MMR ===")
        for i, sentence in enumerate(summary, 1):
            print(f"{i}. {sentence.strip()}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 