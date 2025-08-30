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
df_automl_predictions = pd.DataFrame()
automl_model_info_str = "El modelo de AutoML no se pudo cargar. Ejecuta python_train.py."

try:
    df_roberta = pd.read_csv(PROCESSED_DATA_PATH)
except Exception as e:
    print(f"Advertencia: No se pudo cargar '{PROCESSED_DATA_PATH}'. {e}")

try:
    automl_model_pipeline = load_model(AUTOML_MODEL_PATH)
    automl_model_info_str = f"Mejor modelo encontrado por AutoML: <pre>{str(automl_model_pipeline.steps[-1][1])}</pre>"
    
    if not df_roberta.empty:
        print("Generando predicciones con el modelo AutoML...")
        data_for_automl = df_roberta[['text', 'brand']]
        predictions_automl = predict_model(automl_model_pipeline, data=data_for_automl)
        
        # Unimos las predicciones de AutoML con las emociones detectadas por RoBERTa para una comparación completa
        df_automl_predictions = predictions_automl.rename(columns={'prediction_label': 'sentiment'})
        df_automl_predictions['emotion'] = df_roberta['emotion'] # Añadir columna de emoción
        print("Predicciones de AutoML generadas y enriquecidas.")

except Exception as e:
    print(f"Advertencia: No se pudo cargar o usar el modelo AutoML. {e}")

# --- 3. Funciones de la App ---
def create_sentiment_plot(data, brand_name, model_type):
    if data.empty or 'sentiment' not in data.columns:
        return px.pie(title=f"Sin Datos para {brand_name} ({model_type})")
    sentiment_counts = data['sentiment'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                 title=f"Sentimiento para {brand_name} ({model_type})", color=sentiment_counts.index,
                 color_discrete_map={'Positive':'#2ca02c', 'Negative':'#d62728', 'Neutral':'#7f7f7f'})
    return fig

def create_emotion_plot(data, brand_name, title_suffix=""):
    if data.empty or 'emotion' not in data.columns:
        return px.bar(title=f"Sin Datos de Emociones para {brand_name}")
    emotion_counts = data['emotion'].value_counts().nlargest(7)
    fig = px.bar(emotion_counts, x=emotion_counts.index, y=emotion_counts.values, 
                 title=f"Top Emociones Detectadas {title_suffix}", labels={'x':'Emoción', 'y':'Cantidad'})
    return fig

def update_dashboard(df_source, brand_filter, sentiment_filter):
    if df_source.empty: return None, None, None, pd.DataFrame()
    
    filtered = df_source.copy()
    if sentiment_filter != "Todos": filtered = filtered[filtered['sentiment'] == sentiment_filter]

    if brand_filter == "Ambas":
        intel_df = filtered[filtered['brand'] == 'Intel']
        amd_df = filtered[filtered['brand'] == 'AMD']
        intel_plot = create_sentiment_plot(intel_df, "Intel", "RoBERTa" if 'emotion' in df_source.columns else "AutoML")
        amd_plot = create_sentiment_plot(amd_df, "AMD", "RoBERTa" if 'emotion' in df_source.columns else "AutoML")
        emotion_plot = create_emotion_plot(filtered, "Ambas Marcas", "(RoBERTa)")
        
        return intel_plot, amd_plot, emotion_plot, filtered.head(100)
    else:
        brand_df = filtered[filtered['brand'] == brand_filter]
        brand_plot = create_sentiment_plot(brand_df, brand_filter, "RoBERTa" if 'emotion' in df_source.columns else "AutoML")
        emotion_plot = create_emotion_plot(brand_df, brand_filter, f"({brand_filter} - RoBERTa)")
        return brand_plot, brand_plot, emotion_plot, brand_df.head(100)

# --- 4. Interfaz de Gradio ---
custom_theme = gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]).set(
    body_background_fill="#f0f2f5", block_background_fill="white", block_border_width="1px",
    block_shadow="*shadow_drop_lg", button_primary_background_fill="#00a3c0",
    button_primary_background_fill_hover="#00839c", button_primary_text_color="white",
)
custom_css = "h1 { color: #2c3e50; text-align: center; font-weight: 700; } h3 { color: #333d4d; font-weight: 600; } p { color: #5d6776; text-align: center; }"

with gr.Blocks(theme=custom_theme, css=custom_css, title="Dashboard Comparativo: Intel vs AMD") as demo:
    gr.Markdown("<h1>Dashboard Comparativo: Intel vs AMD en Reddit</h1>")

    with gr.Tabs():
        # --- Pestaña 1: Resultados con AutoML (Nuestro Modelo) ---
        with gr.TabItem("Resultados con AutoML (Nuestro Modelo)"):
            gr.Markdown("<p>Análisis utilizando nuestro modelo propio, entrenado con AutoML. Permite comparar su rendimiento con el modelo experto.</p>")
            with gr.Row():
                brand_filter_automl = gr.Dropdown(choices=["Ambas", "Intel", "AMD"], value="Ambas", label="Filtrar por Marca")
                sentiment_filter_automl = gr.Dropdown(choices=["Todos"] + (df_automl_predictions['sentiment'].unique().tolist() if not df_automl_predictions.empty else []), value="Todos", label="Filtrar por Sentimiento")
            with gr.Row():
                intel_automl_plot = gr.Plot()
                amd_automl_plot = gr.Plot()
            with gr.Row():
                emotion_automl_plot = gr.Plot()
            gr.Markdown("<h3>Vista Previa de Comentarios</h3>")
            comments_automl = gr.DataFrame(headers=["Marca", "Comentario", "Sentimiento (AutoML)", "Emoción (RoBERTa)"], wrap=True)
            
            filters_automl = [brand_filter_automl, sentiment_filter_automl]
            outputs_automl = [intel_automl_plot, amd_automl_plot, emotion_automl_plot, comments_automl]
            demo.load(lambda b, s: update_dashboard(df_automl_predictions, b, s), inputs=filters_automl, outputs=outputs_automl)
            for f in filters_automl: f.change(lambda b, s: update_dashboard(df_automl_predictions, b, s), inputs=filters_automl, outputs=outputs_automl)
            
            gr.Markdown("<h3>Detalles del Modelo AutoML</h3>")
            gr.HTML(value=automl_model_info_str)

        # --- Pestaña 2: Resultados con Modelo Experto (RoBERTa) ---
        with gr.TabItem("Resultados con Modelo Experto (RoBERTa)"):
            gr.Markdown("<p>Análisis de alta precisión utilizando un modelo Transformer pre-entrenado (State-of-the-Art).</p>")
            with gr.Row():
                brand_filter_roberta = gr.Dropdown(choices=["Ambas", "Intel", "AMD"], value="Ambas", label="Filtrar por Marca")
                sentiment_filter_roberta = gr.Dropdown(choices=["Todos"] + (df_roberta['sentiment'].unique().tolist() if not df_roberta.empty else []), value="Todos", label="Filtrar por Sentimiento")
            with gr.Row():
                intel_sentiment_plot = gr.Plot()
                amd_sentiment_plot = gr.Plot()
            with gr.Row():
                emotion_plot_roberta = gr.Plot()
            gr.Markdown("<h3>Vista Previa de Comentarios</h3>")
            comments_roberta = gr.DataFrame(headers=["Marca", "Comentario", "Sentimiento (RoBERTa)", "Emoción"], wrap=True)
            
            filters_roberta = [brand_filter_roberta, sentiment_filter_roberta]
            outputs_roberta = [intel_sentiment_plot, amd_sentiment_plot, emotion_plot_roberta, comments_roberta]
            demo.load(lambda b, s: update_dashboard(df_roberta, b, s), inputs=filters_roberta, outputs=outputs_roberta)
            for f in filters_roberta: f.change(lambda b, s: update_dashboard(df_roberta, b, s), inputs=filters_roberta, outputs=outputs_roberta)

demo.launch()