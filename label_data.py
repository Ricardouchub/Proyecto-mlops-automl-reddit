import pandas as pd
from transformers import pipeline
import torch

# --- Configuración ---
INPUT_CSV = "reddit_comments.csv"
OUTPUT_CSV = "labeled_reddit_comments.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def label_sentiment(df):
    """
    Usa un modelo de Hugging Face para etiquetar el sentimiento de cada comentario.
    """
    print("Cargando el modelo RoBERTa para etiquetado...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)
    
    print(f"Etiquetando {len(df)} comentarios...")
    
    if 'text' not in df.columns:
        print("Error: El CSV debe tener una columna llamada 'text'.")
        return None

    comment_list = df['text'].dropna().astype(str).tolist()

    # --- ¡AQUÍ ESTÁ LA CORRECCIÓN DEFINITIVA! ---
    # Procesamos los comentarios indicando que recorte (truncation) y el límite (max_length)
    results = classifier(comment_list, truncation=True, max_length=512)
    
    # Mapeamos las etiquetas (Positive, Negative, Neutral)
    sentiment_map = {text: res['label'].capitalize() for text, res in zip(comment_list, results)}
    df['sentiment'] = df['text'].map(sentiment_map)
    
    print("Etiquetado completado.")
    return df

# --- Ejecución Principal ---
if __name__ == "__main__":
    try:
        print(f"Leyendo datos desde '{INPUT_CSV}'...")
        comments_df = pd.read_csv(INPUT_CSV)
        
        labeled_df = label_sentiment(comments_df)
        
        if labeled_df is not None:
            print(f"Guardando datos etiquetados de alta calidad en '{OUTPUT_CSV}'...")
            labeled_df.to_csv(OUTPUT_CSV, index=False)
            print("¡Proceso finalizado con éxito!")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{INPUT_CSV}'.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")