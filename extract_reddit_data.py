import praw
import os
import pandas as pd
from dotenv import load_dotenv
import datetime

# Cargar las variables de entorno
load_dotenv()

# --- Configuración ---
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

SEARCH_TERMS = {
    'Intel': ['Intel', 'Intel Arc'],
    'AMD': ['AMD', 'Ryzen', 'Radeon']
}

SUBREDDITS = ['hardware', 'buildapc', 'pcmasterrace', 'AMD', 'Intel']
POST_LIMIT = 50

# --- ¡CAMBIO! Se define la nueva ruta de guardado ---
OUTPUT_FOLDER = "data"
OUTPUT_CSV_NAME = "reddit_comments.csv"
OUTPUT_CSV_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_CSV_NAME)

# --- Lógica del Script (sin cambios en la extracción) ---
def fetch_reddit_data():
    print("Conectando a la API de Reddit...")
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
        )
        print("Conexión exitosa.")
    except Exception as e:
        print(f"Error al conectar a Reddit: {e}")
        return None

    all_comments = []
    
    for brand, keywords in SEARCH_TERMS.items():
        search_query = ' OR '.join(keywords)
        print(f"\nBuscando posts para la marca: '{brand}' con la consulta: '{search_query}'...")

        for sub in SUBREDDITS:
            try:
                subreddit = reddit.subreddit(sub)
                hot_posts = subreddit.search(search_query, limit=POST_LIMIT, sort='hot')
                
                for post in hot_posts:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        cleaned_text = comment.body.replace('\n', ' ').replace('\r', ' ').strip()
                        
                        if cleaned_text and cleaned_text != '[deleted]' and cleaned_text != '[removed]':
                            all_comments.append({
                                'brand': brand,
                                'comment_id': comment.id,
                                'text': cleaned_text,
                                'subreddit': sub,
                                'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
                                'score': comment.score
                            })
            except Exception as e:
                print(f"   - No se pudo procesar el subreddit '{sub}': {e}")

    print(f"\nSe encontraron {len(all_comments)} comentarios en total.")
    if not all_comments:
        return None
        
    return pd.DataFrame(all_comments)

def save_data(df):
    print(f"Guardando {len(df)} comentarios en '{OUTPUT_CSV_PATH}'...")
    # Sobrescribimos el archivo cada vez para tener los datos más frescos
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Archivo guardado con éxito.")

# --- Ejecución Principal ---
if __name__ == "__main__":
    # --- ¡CAMBIO! Asegurarse de que la carpeta de datos exista ---
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Carpeta '{OUTPUT_FOLDER}' creada.")

    # Eliminar el archivo viejo para empezar de cero
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)
        print(f"Archivo antiguo '{OUTPUT_CSV_PATH}' eliminado.")
        
    new_data_df = fetch_reddit_data()
    if new_data_df is not None and not new_data_df.empty:
        save_data(new_data_df)