import gradio as gr
import pandas as pd
import plotly.express as px
import os

# --- 1. Definición de Rutas ---
DATA_FOLDER = "data"
MODELS_FOLDER = "models"
AUTOML_RESULTS_PATH = os.path.join(DATA_FOLDER, "automl_results.csv")
MODEL_INFO_PATH = os.path.join(MODELS_FOLDER, "model_info.txt")

# --- 2. Carga de Datos y Metadatos (Súper Ligero) ---
# Esta app NO CARGA el modelo .pkl, solo sus resultados pre-calculados.
df_automl_predictions = pd.DataFrame()
automl_model_info_str = "La información del modelo no se encontró. Asegúrate de que el pipeline de entrenamiento se haya ejecutado."

try:
    # Cargar los resultados PRE-CALCULADOS por el pipeline de entrenamiento
    df_automl_predictions = pd.read_csv(AUTOML_RESULTS_PATH)
    print("Archivo de resultados de AutoML cargado correctamente.")
except Exception as e:
    print(f"Error al cargar '{AUTOML_RESULTS_PATH}': {e}")
    # Crear un DataFrame vacío con las columnas esperadas para que la app no falle
    df_automl_predictions = pd.DataFrame(columns=['brand', 'text', 'sentiment'])

try:
    # Cargar la descripción del modelo desde un archivo de texto simple
    with open(MODEL_INFO_PATH, "r") as f:
        automl_model_info_str = f.read()
except Exception as e:
    print(f"Advertencia: No se pudo cargar la información del modelo desde '{MODEL_INFO_PATH}': {e}")

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
    """
    Filtra el DataFrame ya cargado en memoria y actualiza los componentes de la UI.
    """
    if df_automl_predictions.empty: 
        return None, None, pd.DataFrame()
    
    filtered = df_automl_predictions.copy()
    if brand_filter != "Ambas": 
        filtered = filtered[filtered['brand'] == brand_filter]
    if sentiment_filter != "Todos": 
        filtered = filtered[filtered['sentiment'] == sentiment_filter]

    if brand_filter == "Ambas":
        intel_df = filtered[filtered['brand'] == 'Intel']
        amd_df = filtered[filtered['brand'] == 'AMD']
        intel_plot = create_sentiment_plot(intel_df, "Intel")
        amd_plot = create_sentiment_plot(amd_df, "AMD")
        return intel_plot, amd_plot, filtered[['brand', 'text', 'sentiment']].head(100)
    else: # Si se selecciona una marca específica
        brand_df = filtered[filtered['brand'] == brand_filter]
        brand_plot = create_sentiment_plot(brand_df, brand_filter)
        # Devolvemos el mismo gráfico dos veces para llenar ambos espacios de la UI
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
    
    # Conectar los filtros a la función de actualización
    filters = [brand_filter, sentiment_filter]
    outputs = [intel_plot, amd_plot, comments_table]
    demo.load(fn=update_dashboard, inputs=filters, outputs=outputs)
    for f in filters: 
        f.change(fn=update_dashboard, inputs=filters, outputs=outputs)

# Lanzar la aplicación para el despliegue en Render
demo.launch(server_name="0.0.0.0", server_port=7860)