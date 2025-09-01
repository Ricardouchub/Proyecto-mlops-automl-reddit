import gradio as gr
import pandas as pd
import plotly.express as px
from pycaret.classification import load_model, predict_model
import os
import ast

# --- 1. Definición de Rutas ---
DATA_FOLDER = "data"
MODELS_FOLDER = "models"
PROCESSED_DATA_PATH = os.path.join(DATA_FOLDER, "processed_reddit_data.csv")
AUTOML_MODEL_PATH = os.path.join(MODELS_FOLDER, "sentiment_model_v2")

# --- 2. Carga de Datos y Modelos ---
df_roberta = pd.DataFrame()
automl_model_pipeline = None
automl_model_info_str = "El modelo de AutoML no se pudo cargar. Ejecuta python_train.py."

try:
    df_roberta = pd.read_csv(PROCESSED_DATA_PATH)
except Exception as e:
    print(f"Advertencia: No se pudo cargar '{PROCESSED_DATA_PATH}'. {e}")

try:
    automl_model_pipeline = load_model(AUTOML_MODEL_PATH)
    automl_model_info_str = f"Mejor modelo encontrado por AutoML: <pre>{str(automl_model_pipeline.steps[-1][1])}</pre>"
except Exception as e:
    print(f"Advertencia: No se pudo cargar el modelo AutoML. {e}")

# --- 3. Funciones de la App ---
def create_sentiment_plot(data, brand_name, model_type):
    if data.empty or 'sentiment' not in data.columns:
        return px.pie(title=f"Sin Datos para {brand_name} ({model_type})")
    sentiment_counts = data['sentiment'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                 title=f"Sentimiento para {brand_name} ({model_type})", color=sentiment_counts.index,
                 color_discrete_map={'Positive':'#2ca02c', 'Negative':'#d62728', 'Neutral':'#7f7f7f'})
    return fig

def create_emotion_plot(data, brand_name):
    if data.empty or 'emotion' not in data.columns: return px.bar(title="Sin Datos de Emociones")
    emotion_counts = data['emotion'].value_counts().nlargest(7)
    return px.bar(emotion_counts, x=emotion_counts.index, y=emotion_counts.values, 
                  title=f"Top Emociones Detectadas ({brand_name})", labels={'x':'Emoción', 'y':'Cantidad'})

def update_roberta_dashboard(brand_filter, sentiment_filter):
    if df_roberta.empty: return None, None, None, pd.DataFrame()
    filtered = df_roberta.copy()
    if sentiment_filter != "Todos": filtered = filtered[filtered['sentiment'] == sentiment_filter]

    if brand_filter == "Ambas":
        intel_df = filtered[filtered['brand'] == 'Intel']
        amd_df = filtered[filtered['brand'] == 'AMD']
        intel_plot = create_sentiment_plot(intel_df, "Intel", "RoBERTa")
        amd_plot = create_sentiment_plot(amd_df, "AMD", "RoBERTa")
        emotion_plot = create_emotion_plot(filtered, "Ambas Marcas")
        return intel_plot, amd_plot, emotion_plot, filtered[['brand', 'text', 'sentiment', 'emotion']].head(100)
    else:
        brand_df = filtered[filtered['brand'] == brand_filter]
        brand_plot = create_sentiment_plot(brand_df, brand_filter, "RoBERTa")
        emotion_plot = create_emotion_plot(brand_df, brand_filter)
        return brand_plot, brand_plot, emotion_plot, brand_df[['brand', 'text', 'sentiment', 'emotion']].head(100)

# --- ¡CAMBIO CLAVE! ---
# Nueva función que se ejecuta SOLO cuando se hace clic en la pestaña de AutoML
def generate_and_update_automl_dashboard(brand_filter):
    if automl_model_pipeline is None or df_roberta.empty:
        return None, None, pd.DataFrame()

    print("Generando predicciones de AutoML bajo demanda...")
    # Se generan las predicciones en el momento
    predictions_automl = predict_model(automl_model_pipeline, data=df_roberta[['text', 'brand']])
    df_automl_predictions = predictions_automl.rename(columns={'prediction_label': 'sentiment'})
    print("Predicciones de AutoML generadas.")

    filtered = df_automl_predictions.copy()
    if brand_filter != "Ambas":
        filtered = filtered[filtered['brand'] == brand_filter]

    if brand_filter == "Ambas":
        intel_df = filtered[filtered['brand'] == 'Intel']
        amd_df = filtered[filtered['brand'] == 'AMD']
        intel_plot = create_sentiment_plot(intel_df, "Intel", "AutoML")
        amd_plot = create_sentiment_plot(amd_df, "AMD", "AutoML")
        return intel_plot, amd_plot, filtered[['brand', 'text', 'sentiment']].head(100)
    else:
        brand_df = filtered[filtered['brand'] == brand_filter]
        brand_plot = create_sentiment_plot(brand_df, brand_filter, "AutoML")
        return brand_plot, brand_plot, brand_df[['brand', 'text', 'sentiment']].head(100)

# --- 4. Interfaz de Gradio ---
custom_theme = gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]).set(
    body_background_fill="#f0f2f5", block_background_fill="white", block_border_width="1px",
    block_shadow="*shadow_drop_lg", button_primary_background_fill="#00a3c0",
    button_primary_background_fill_hover="#00839c", button_primary_text_color="white",
)
custom_css = "h1 { color: #2c3e50; text-align: center; font-weight: 700; } h3 { color: #333d4d; font-weight: 600; } p { color: #5d6776; text-align: center; }"

with gr.Blocks(theme=custom_theme, css=custom_css, title="Dashboard Comparativo: Intel vs AMD") as demo:
    gr.Markdown("<h1>📊 Dashboard Comparativo: Intel vs AMD en Reddit</h1>")

    with gr.Tabs() as tabs:
        with gr.TabItem("Resultados con Modelo Experto (RoBERTa)", id=0):
            gr.Markdown("<p>Análisis de alta precisión utilizando un modelo Transformer pre-entrenado.</p>")
            with gr.Row():
                brand_filter_roberta = gr.Dropdown(choices=["Ambas", "Intel", "AMD"], value="Ambas", label="Filtrar por Marca")
                sentiment_filter_roberta = gr.Dropdown(choices=["Todos"] + (df_roberta['sentiment'].unique().tolist() if not df_roberta.empty else []), value="Todos", label="Filtrar por Sentimiento")
            with gr.Row():
                intel_sentiment_plot = gr.Plot()
                amd_sentiment_plot = gr.Plot()
            with gr.Row():
                emotion_plot = gr.Plot()
            gr.Markdown("<h3>Vista Previa de Comentarios</h3>")
            comments_roberta = gr.DataFrame(headers=["Marca", "Comentario", "Sentimiento (RoBERTa)", "Emoción"], wrap=True)
            
            filters_roberta = [brand_filter_roberta, sentiment_filter_roberta]
            outputs_roberta = [intel_sentiment_plot, amd_sentiment_plot, emotion_plot, comments_roberta]
            demo.load(fn=update_roberta_dashboard, inputs=filters_roberta, outputs=outputs_roberta)
            for f in filters_roberta: f.change(fn=update_roberta_dashboard, inputs=filters_roberta, outputs=outputs_roberta)

        with gr.TabItem("Resultados con AutoML (Nuestro Modelo)", id=1):
            gr.Markdown("<p>Análisis utilizando nuestro modelo propio, entrenado con AutoML.</p>")
            brand_filter_automl = gr.Dropdown(choices=["Ambas", "Intel", "AMD"], value="Ambas", label="Filtrar por Marca")
            with gr.Row():
                intel_automl_plot = gr.Plot()
                amd_automl_plot = gr.Plot()
            gr.Markdown("<h3>Vista Previa de Comentarios</h3>")
            comments_automl = gr.DataFrame(headers=["Marca", "Comentario", "Sentimiento (AutoML)"], wrap=True)
            gr.Markdown("<h3>Detalles del Modelo AutoML</h3>")
            gr.HTML(value=automl_model_info_str)
            
            # --- ¡CAMBIO CLAVE! ---
            # Los gráficos se actualizan cuando se selecciona la pestaña o cuando cambia el filtro
            outputs_automl = [intel_automl_plot, amd_automl_plot, comments_automl]
            tabs.select(lambda tab_id: generate_and_update_automl_dashboard("Ambas") if tab_id == 1 else gr.skip(), inputs=tabs, outputs=outputs_automl)
            brand_filter_automl.change(fn=generate_and_update_automl_dashboard, inputs=brand_filter_automl, outputs=outputs_automl)

# Ajuste final para el despliegue en Render
demo.launch(server_name="0.0.0.0", server_port=7860)