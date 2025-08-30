import pandas as pd
from pycaret.classification import *
import os
import torch # Importamos torch para verificar la disponibilidad de la GPU

# --- 1. Definición de Rutas ---
DATA_FOLDER = "data"
MODELS_FOLDER = "models"
INPUT_CSV = os.path.join(DATA_FOLDER, "processed_reddit_data.csv")
MODEL_NAME = "sentiment_model_v2"
MODEL_PATH = os.path.join(MODELS_FOLDER, MODEL_NAME)
PLOT_FILENAME = "Feature Importance.png"

# --- 2. Función Principal de Entrenamiento ---
def train_model():
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
        print(f"Carpeta '{MODELS_FOLDER}' creada.")

    print(f"Cargando datos desde '{INPUT_CSV}'...")
    dataset = pd.read_csv(INPUT_CSV)
    dataset.dropna(subset=['text', 'sentiment'], inplace=True)
    print(f"Datos cargados. Total de comentarios para entrenar: {len(dataset)}")

    # --- ¡CAMBIO CLAVE! ---
    # Detectar si hay una GPU disponible para PyCaret
    gpu_enabled = torch.cuda.is_available()
    if gpu_enabled:
        print("GPU detectada. Se intentará usar para el entrenamiento de AutoML.")
    else:
        print("No se detectó GPU. Se usará la CPU para el entrenamiento.")
    # --- FIN DEL CAMBIO ---
    
    ignore_features = ['comment_id', 'subreddit', 'created_utc', 'score', 'emotion', 'entities', 'topic']
    
    print("Configurando el experimento de clasificación para NLP...")
    s = setup(data=dataset,
              target='sentiment',
              text_features=['text'],
              ignore_features=ignore_features,
              session_id=123,
              log_experiment=True,
              experiment_name="reddit_sentiment_analysis_intel_amd",
              use_gpu=gpu_enabled, # ¡CAMBIO CLAVE! Se activa la GPU si está disponible
              verbose=False)
    print("Configuración completa.")

    print("Buscando el mejor modelo entre [LightGBM, Random Forest, Extra Trees]...")
    best_model = compare_models(include=['lightgbm', 'rf', 'et'])
    print("\nMejor modelo encontrado:")
    print(best_model)

    print(f"\nGuardando gráfico de importancia como '{PLOT_FILENAME}'...")
    try:
        plot_model(best_model, plot='feature', save=True)
        generated_files = [f for f in os.listdir() if f.endswith('_feature.png')]
        if generated_files:
            if os.path.exists(PLOT_FILENAME): os.remove(PLOT_FILENAME)
            os.rename(generated_files[0], PLOT_FILENAME)
            print("Gráfico guardado exitosamente.")
        else:
            print("Advertencia: No se encontró el archivo del gráfico.")
    except Exception as e:
        print(f"Error al generar o guardar el gráfico: {e}")

    print("\nFinalizando y guardando el modelo...")
    final_model = finalize_model(best_model)
    save_model(final_model, MODEL_PATH)
    print(f"¡Modelo guardado exitosamente como '{MODEL_PATH}.pkl'!")

if __name__ == "__main__":
    train_model()