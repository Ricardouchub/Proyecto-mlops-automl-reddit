import pandas as pd
from transformers import pipeline
import torch
import re
import nltk
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import os

DATA_FOLDER = "data"

try:
    stopwords.words('english')
except LookupError:
    print("Descargando recursos de NLTK (stopwords, wordnet)...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("Recursos descargados.")

# Configuración de Modelos
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
NER_MODEL = "dslim/bert-base-NER"

# Funciones de Análisis (sin cambios en su lógica interna)
def analyze_sentiment_emotion(df):
    print("1. Analizando Sentimiento y Emociones...")
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("   - GPU detectada. Usando CUDA para la aceleración.")
    else:
        print("   - No se detectó GPU. Usando CPU.")
    sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=device)
    emotion_pipeline = pipeline("text-classification", model=EMOTION_MODEL, device=device)

    comments = df['text'].dropna().astype(str).tolist()
    
    # Sentimiento
    sentiments = sentiment_pipeline(comments, truncation=True, max_length=512)
    sentiment_map = {text: res['label'].capitalize() for text, res in zip(comments, sentiments)}
    df['sentiment'] = df['text'].map(sentiment_map)
    
    # Emociones
    emotions = emotion_pipeline(comments, truncation=True, max_length=512)
    emotion_map = {text: res['label'].capitalize() for text, res in zip(comments, emotions)}
    df['emotion'] = df['text'].map(emotion_map)
    
    print("   ...Análisis de Sentimiento y Emociones completado.")
    return df

def extract_entities(df):
    print("2. Extrayendo Entidades Nombradas (NER)...")
    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = pipeline("ner", model=NER_MODEL, device=device, aggregation_strategy="simple")

    comments = df['text'].dropna().astype(str).tolist()
    
    all_entities = []
    batch_size = 50
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        results = ner_pipeline(batch)
        all_entities.extend(results)

    entity_map = {text: entities for text, entities in zip(comments, all_entities)}
    df['entities'] = df['text'].map(entity_map)

    print("   ...Extracción de Entidades completada.")
    return df

def discover_topics(df):
    print("3. Descubriendo Tópicos (Topic Modeling)...")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        tokens = text.split()
        return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 3]

    processed_docs = df['text'].dropna().astype(str).map(preprocess)
    
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15, random_state=42)
    
    topic_map = {}
    for i, doc in enumerate(processed_docs):
        original_text = df['text'].iloc[i]
        bow = dictionary.doc2bow(doc)
        topics = lda_model.get_document_topics(bow)
        if topics:
            dominant_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0][0]
            topic_map[original_text] = f"Tópico {dominant_topic + 1}"
        else:
            topic_map[original_text] = "Indefinido"

    df['topic'] = df['text'].map(topic_map)
    
    topic_keywords = {f"Tópico {i+1}": ", ".join([word for word, _ in lda_model.show_topic(i, topn=5)]) for i in range(lda_model.num_topics)}
    
    print("   ...Modelado de Tópicos completado.")
    return df, topic_keywords

# Lógica Principal del Pipeline
if __name__ == "__main__":
    print("Iniciando pipeline de procesamiento de NLP...")
    
    input_path = os.path.join(DATA_FOLDER, "reddit_comments.csv")
    output_processed_path = os.path.join(DATA_FOLDER, "processed_reddit_data.csv")
    output_topics_path = os.path.join(DATA_FOLDER, "topic_keywords.csv")

    df = pd.read_csv(input_path)
    
    df = analyze_sentiment_emotion(df)
    df = extract_entities(df)
    df, topic_keywords = discover_topics(df)
    
    df.to_csv(output_processed_path, index=False)
    pd.DataFrame(list(topic_keywords.items()), columns=['Topic', 'Keywords']).to_csv(output_topics_path, index=False)

    print(f"\n¡Pipeline de procesamiento finalizado con éxito!")
    print(f"Archivos procesados guardados en la carpeta '{DATA_FOLDER}'.")