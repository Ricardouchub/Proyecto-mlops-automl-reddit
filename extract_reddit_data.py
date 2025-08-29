import praw
import os
import pandas as pd
from dotenv import load_dotenv
import datetime

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# --- Configuración ---
# Lee las credenciales de la API desde las variables de entorno
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Palabras clave que queremos buscar
KEYWORDS = ['bitcoin', 'btc']

# Subreddits donde queremos buscar
SUBREDDITS = ['CryptoCurrency', 'Bitcoin', 'CryptoMarkets', 'ethtrader', 'SatoshiStreetBets']

# Número máximo de posts a revisar en cada subreddit
POST_LIMIT = 20 # Aumenta este número para obtener más datos, pero sé consciente de los límites de la API

# Nombre del archivo de salida
OUTPUT_CSV = "reddit_comments.csv"

# --- Lógica del Script ---
def fetch_reddit_data():
    """
    Se conecta a la API de Reddit, busca posts con las palabras clave
    en los subreddits especificados y extrae los comentarios.
    """
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
    search_query = ' OR '.join(KEYWORDS)

    print(f"Buscando posts con '{search_query}' en {len(SUBREDDITS)} subreddits...")

    for sub in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub)
            # Usamos search para encontrar posts relevantes
            hot_posts = subreddit.search(search_query, limit=POST_LIMIT, sort='hot')

            for post in hot_posts:
                # Cargar todos los comentarios del post (puede tardar)
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    # Limpiar el texto del comentario
                    cleaned_text = comment.body.replace('\n', ' ').replace('\r', ' ').strip()

                    # Solo añadimos comentarios con texto
                    if cleaned_text and cleaned_text != '[deleted]' and cleaned_text != '[removed]':
                        all_comments.append({
                            'comment_id': comment.id,
                            'text': cleaned_text,
                            'subreddit': sub,
                            'created_utc': datetime.datetime.utcfromtimestamp(comment.created_utc),
                            'score': comment.score
                        })
        except Exception as e:
            print(f"No se pudo procesar el subreddit '{sub}': {e}")

    print(f"Se encontraron {len(all_comments)} comentarios en total.")
    if not all_comments:
        print("No se encontraron nuevos comentarios. Terminando.")
        return None

    return pd.DataFrame(all_comments)

def save_data(df):
    """
    Guarda el DataFrame en un archivo CSV.
    Si el archivo ya existe, añade los nuevos datos sin duplicados.
    """
    if os.path.exists(OUTPUT_CSV):
        print(f"El archivo '{OUTPUT_CSV}' ya existe. Añadiendo nuevos comentarios...")
        existing_df = pd.read_csv(OUTPUT_CSV)
        combined_df = pd.concat([existing_df, df])
        # Eliminar duplicados por si volvemos a capturar el mismo comentario
        combined_df.drop_duplicates(subset=['comment_id'], keep='last', inplace=True)
        combined_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Archivo actualizado. Total de comentarios ahora: {len(combined_df)}")
    else:
        print(f"Creando nuevo archivo '{OUTPUT_CSV}'...")
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Archivo guardado con {len(df)} comentarios.")

# --- Ejecución Principal ---
if __name__ == "__main__":
    new_data_df = fetch_reddit_data()
    if new_data_df is not None and not new_data_df.empty:
        save_data(new_data_df)