import gradio as gr
import pandas as pd
import plotly.express as px
from pycaret.classification import load_model, predict_model
import os

# --- 1. Definición de Rutas ---
DATA_FOLDER = "data"
MODELS_FOLDER = "models"
PROCESSED_DATA_PATH = os.path.join(DATA_FOLDER, "processed_reddit_data.csv")
AUTOML_MODEL_PATH = os.path.join(MODELS_FOLDER, "sentiment_model_v2")

# --- 2. Carga de Datos y Modelo AutoML ---
df_automl_predictions = pd.DataFrame()
automl_model_info_str = "El modelo de AutoML no se pudo cargar. Asegúrate de que el pipeline de entrenamiento se haya ejecutado con éxito."

try:
    automl_model_pipeline = load_model(AUTOML_MODEL_PATH)
    automl_model_info_str = f"Mejor modelo encontrado por AutoML: <pre>{str(automl_model_pipeline.steps[-1][1])}</pre>"
    
    df_processed = pd.read_csv(PROCESSED_DATA_PATH)
    
    if not df_processed.empty:
        print("Generando predicciones con el modelo AutoML...")
        
        # --- ¡CORRECCIÓN CLAVE! ---
        # Creamos un DataFrame limpio solo con las columnas que el modelo fue entrenado.
        data_for_prediction = df_processed[['text', 'brand']]
        
        # Le pasamos solo esos datos a la función de predicción.
        predictions = predict_model(automl_model_pipeline, data=data_for_prediction)
        
        # Unimos las predicciones con la marca para poder filtrar en el dashboard.
        df_automl_predictions = pd.concat([
            df_processed['brand'].reset_index(drop=True),
            predictions[['prediction_label', 'prediction_score', 'text']].reset_index(drop=True)
        ], axis=1)
        df_automl_predictions = df_automl_predictions.rename(columns={'prediction_label': 'sentiment'})
        print("Predicciones de AutoML generadas correctamente.")

except Exception as e:
    print(f"Advertencia: Ocurrió un error al cargar o usar el modelo AutoML. {e}")

# --- 3. Funciones de la App ---
def create_sentiment_plot(data, brand_name):
    if data.empty or 'sentiment' not in data.columns:
        return px.pie(title=f"Sin Datos de Sentimiento para {brand_name}")
    sentiment_counts = data['sentiment'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                 title=f"Sentimiento para {brand_name} (Modelo AutoML)", color=sentiment_counts.index,
                 color_discrete_map={'Positive':'#2ca02c', 'Negative':'#d62728', 'Neutral':'#7f7f7f'})
    return fig

def update_dashboard(brand_filter, sentiment_filter):
    if df_automl_predictions.empty: return None, None, pd.DataFrame()
    
    filtered = df_automl_predictions.copy()
    if brand_filter != "Ambas": filtered = filtered[filtered['brand'] == brand_filter]
    if sentiment_filter != "Todos": filtered = filtered[filtered['sentiment'] == sentiment_filter]

    if brand_filter == "Ambas":
        intel_df = filtered[filtered['brand'] == 'Intel']
        amd_df = filtered[filtered['brand'] == 'AMD']
        intel_plot = create_sentiment_plot(intel_df, "Intel")
        amd_plot = create_sentiment_plot(amd_df, "AMD")
        return intel_plot, amd_plot, filtered[['brand', 'text', 'sentiment']].head(100)
    else:
        brand_df = filtered[filtered['brand'] == brand_filter]
        brand_plot = create_sentiment_plot(brand_df, brand_filter)
        return brand_plot, brand_plot, brand_df[['brand', 'text', 'sentiment']].head(100)

# --- 4. Interfaz de Gradio ---
custom_theme = gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]).set(
    body_background_fill="#f0f2f5", block_background_fill="white", block_border_width="1px",
    block_shadow="*shadow_drop_lg", button_primary_background_fill="#00a3c0",
    button_primary_background_fill_hover="#00839c", button_primary_text_color="white",
)
custom_css = "h1 { color: #2c3e50; text-align: center; font-weight: 700; } h3 { color: #333d4d; font-weight: 600; } p { color: #5d6776; text-align: center; }"

with gr.Blocks(theme=custom_theme, css=custom_css, title="Dashboard de AutoML: Intel vs AMD") as demo:
    gr.Markdown("<h1>📊 Dashboard de Resultados del Pipeline de AutoML</h1>")
    gr.Markdown("<p>Análisis de sentimiento utilizando nuestro modelo propio, entrenado y desplegado automáticamente con un pipeline de MLOps.</p>")
    
    with gr.Row():
        brand_filter = gr.Dropdown(choices=["Ambas", "Intel", "AMD"], value="Ambas", label="Filtrar por Marca")
        sentiment_filter = gr.Dropdown(choices=["Todos"] + (df_automl_predictions['sentiment'].unique().tolist() if not df_automl_predictions.empty else []), value="Todos", label="Filtrar por Sentimiento")
    
    with gr.Row():
        intel_plot = gr.Plot()
        amd_plot = gr.Plot()
        
    gr.Markdown("<h3>Vista Previa de Comentarios</h3>")
    comments_table = gr.DataFrame(headers=["Marca", "Comentario", "Sentimiento (AutoML)"], wrap=True)
    
    gr.Markdown("<h3>Detalles del Modelo AutoML</h3>")
    gr.HTML(value=automl_model_info_str)
    
    filters = [brand_filter, sentiment_filter]
    outputs = [intel_plot, amd_plot, comments_table]
    demo.load(fn=update_dashboard, inputs=filters, outputs=outputs)
    for f in filters: f.change(fn=update_dashboard, inputs=filters, outputs=outputs)

# Ajuste para el despliegue en Render
demo.launch(server_name="0.0.0.0", server_port=7860)