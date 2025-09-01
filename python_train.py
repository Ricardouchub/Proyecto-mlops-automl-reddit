import pandas as pd
from pycaret.classification import *
import os

# --- 1. Definición de Rutas ---
DATA_FOLDER = "data"
MODELS_FOLDER = "models"
INPUT_CSV = os.path.join(DATA_FOLDER, "processed_reddit_data.csv")
OUTPUT_CSV = os.path.join(DATA_FOLDER, "automl_results.csv")
MODEL_NAME = "sentiment_model_v2"
MODEL_PATH = os.path.join(MODELS_FOLDER, MODEL_NAME)
PLOT_FILENAME = "Feature Importance.png"
MODEL_INFO_PATH = os.path.join(MODELS_FOLDER, "model_info.txt") # ¡NUEVO!

# --- 2. Función Principal ---
def train_and_predict():
    if not os.path.exists(MODELS_FOLDER): os.makedirs(MODELS_FOLDER)

    print(f"Cargando datos desde '{INPUT_CSV}'...")
    dataset = pd.read_csv(INPUT_CSV)
    dataset.dropna(subset=['text', 'sentiment'], inplace=True)
    
    ignore_features = ['comment_id', 'subreddit', 'created_utc', 'score', 'emotion', 'entities', 'topic']
    setup(data=dataset, target='sentiment', text_features=['text'],
          ignore_features=ignore_features, session_id=123,
          log_experiment=True, experiment_name="reddit_sentiment_analysis_intel_amd",
          verbose=False)
    
    print("Buscando el mejor modelo...")
    best_model = compare_models(include=['lightgbm', 'rf', 'et'])
    print("\nMejor modelo encontrado:", best_model)
    
    # Guardar gráfico de importancia
    try:
        plot_model(best_model, plot='feature', save=True)
        generated_files = [f for f in os.listdir() if f.endswith('_feature.png')]
        if generated_files:
            if os.path.exists(PLOT_FILENAME): os.remove(PLOT_FILENAME)
            os.rename(generated_files[0], PLOT_FILENAME)
            print("Gráfico de importancia guardado.")
    except Exception as e:
        print(f"Advertencia: No se pudo generar el gráfico: {e}")

    final_model = finalize_model(best_model)
    save_model(final_model, MODEL_PATH)
    print(f"Modelo guardado en '{MODEL_PATH}.pkl'")
    
    # --- ¡NUEVO! Guardar la descripción del modelo en un .txt ---
    with open(MODEL_INFO_PATH, "w") as f:
        f.write(f"Mejor modelo encontrado por AutoML: <pre>{str(final_model.steps[-1][1])}</pre>")
    print(f"Información del modelo guardada en '{MODEL_INFO_PATH}'")

    print(f"Generando predicciones y guardando en '{OUTPUT_CSV}'...")
    predictions = predict_model(final_model, data=dataset)
    results_df = predictions[['brand', 'text', 'prediction_label']]
    results_df = results_df.rename(columns={'prediction_label': 'sentiment'})
    results_df.to_csv(OUTPUT_CSV, index=False)
    print("Archivo de resultados de AutoML guardado.")

if __name__ == "__main__":
    train_and_predict()