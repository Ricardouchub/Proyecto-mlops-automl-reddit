import os
import pandas as pd
import plotly.express as px
import gradio as gr

# --- 1) Rutas ---
DATA_FOLDER = "data"
MODELS_FOLDER = "models"
AUTOML_RESULTS_PATH = os.path.join(DATA_FOLDER, "automl_results.csv")
MODEL_INFO_PATH = os.path.join(MODELS_FOLDER, "model_info.txt")

# --- 2) Carga de datos (ligero) ---
df_automl_predictions = pd.DataFrame()
try:
    df_automl_predictions = _read_csv_safe(AUTOML_RESULTS_PATH)
    # DEBUG: imprime información clave al log de Render
    print("[DEBUG] CSV path:", os.path.abspath(AUTOML_RESULTS_PATH))
    print("[DEBUG] CSV exists?:", os.path.exists(AUTOML_RESULTS_PATH))
    try:
        print("[DEBUG] CSV size (bytes):", os.path.getsize(AUTOML_RESULTS_PATH))
    except Exception as _e:
        print("[DEBUG] os.path.getsize failed:", _e)
    print("[DEBUG] shape:", df_automl_predictions.shape)
    print("[DEBUG] head(5):\n", df_automl_predictions.head(5).to_string(index=False))
    if set(["brand","sentiment"]).issubset(df_automl_predictions.columns):
        print("[DEBUG] brand counts:\n", df_automl_predictions["brand"].value_counts(dropna=False).to_string())
        print("[DEBUG] sentiment counts:\n", df_automl_predictions["sentiment"].value_counts(dropna=False).to_string())
        print("[DEBUG] brand x sentiment:\n",
              df_automl_predictions.groupby(["brand","sentiment"]).size().reset_index(name="count").to_string(index=False))
    print("Archivo de resultados de AutoML cargado correctamente.")
except Exception as e:
    print(f"Error al cargar '{AUTOML_RESULTS_PATH}': {e}")
    df_automl_predictions = pd.DataFrame(columns=["brand", "text", "sentiment"])

automl_model_info_str = (
    "La información del modelo no se encontró. Asegúrate de que el pipeline de entrenamiento se haya ejecutado."
)

def _read_csv_safe(path: str) -> pd.DataFrame:
    """Lee un CSV intentando con distintas codificaciones."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # Si falla todo, relanza la excepción original de utf-8
    return pd.read_csv(path, encoding="utf-8")

try:
    df_automl_predictions = _read_csv_safe(AUTOML_RESULTS_PATH)
    print("Archivo de resultados de AutoML cargado correctamente.")
except Exception as e:
    print(f"Error al cargar '{AUTOML_RESULTS_PATH}': {e}")
    df_automl_predictions = pd.DataFrame(columns=["brand", "text", "sentiment"])

try:
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        automl_model_info_str = f.read()
except Exception as e:
    print(f"Advertencia: No se pudo cargar '{MODEL_INFO_PATH}': {e}")

# --- 3) Normalización / utilidades ---
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
BRAND_ORDER = ["Intel", "AMD"]

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas clave: quita espacios, corrige capitalización y mapea a etiquetas estándar."""
    if df.empty:
        return df
    df = df.copy()

    if "brand" in df.columns:
        df["brand"] = (
            df["brand"]
            .astype(str).str.strip().str.title()  # "intel" -> "Intel"
            .replace({"Amd": "AMD"})              # title-case convierte "AMD" en "Amd"; devolvemos "AMD"
        )
    if "sentiment" in df.columns:
        df["sentiment"] = (
            df["sentiment"]
            .astype(str).str.strip().str.title()  # "positive" -> "Positive"
        )
        # Mapeos comunes a etiquetas finales
        df["sentiment"] = df["sentiment"].replace({
            "Positivo": "Positive",
            "Negativo": "Negative",
            "Neutralidad": "Neutral",
            "Neutro": "Neutral",
        })

        # Mantener solo etiquetas esperadas si quieres filtrar ruido
        df = df[df["sentiment"].isin(SENTIMENT_ORDER)]

    return df

df_automl_predictions = _normalize_df(df_automl_predictions)

def _prep_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte un df con columna 'sentiment' a un DataFrame con columnas
    ['sentiment', 'count'] ordenado y listo para plotear.
    """
    if df.empty or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["sentiment", "count"])

    counts = (
        df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    # Asegura orden fijo
    counts = counts[counts["sentiment"].isin(SENTIMENT_ORDER)]
    counts["sentiment"] = pd.Categorical(counts["sentiment"], categories=SENTIMENT_ORDER, ordered=True)
    counts = counts.sort_values("sentiment")
    return counts

def _prep_brand_sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta por (brand, sentiment) para la comparación Intel vs AMD."""
    if df.empty or not {"brand", "sentiment"}.issubset(df.columns):
        return pd.DataFrame(columns=["brand", "sentiment", "count"])
    g = (
        df.groupby(["brand", "sentiment"])
          .size()
          .reset_index(name="count")
    )
    # Orden consistente
    g = g[g["sentiment"].isin(SENTIMENT_ORDER)]
    g["sentiment"] = pd.Categorical(g["sentiment"], categories=SENTIMENT_ORDER, ordered=True)
    g = g.sort_values(["sentiment", "brand"])
    return g

# --- 4) Funciones de gráficos ---
def create_sentiment_pie(data: pd.DataFrame, brand_name: str):
    counts = _prep_counts(data)
    if counts.empty:
        return px.pie(title=f"Sin Datos de Sentimiento para {brand_name}")
    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        title=f"Sentimiento para {brand_name} (Modelo AutoML)",
        color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"},
    )
    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value} (%{percent})")
    return fig

def create_compare_bar(data: pd.DataFrame):
    """
    Barra agrupada comparando Intel vs AMD por sentimiento.
    Si no hay datos, muestra un título indicativo.
    """
    g = _prep_brand_sentiment_counts(data)
    if g.empty:
        return px.bar(title="Comparación Intel vs AMD (sin datos)")
    # Asegurar orden deseado de brand
    g["brand"] = pd.Categorical(g["brand"], categories=BRAND_ORDER, ordered=True)
    g = g.sort_values(["sentiment", "brand"])
    fig = px.bar(
        g,
        x="sentiment",
        y="count",
        color="brand",
        barmode="group",
        title="Comparación Intel vs AMD por Sentimiento",
    )
    fig.update_layout(xaxis_title="Sentimiento", yaxis_title="Conteo")
    return fig

# --- 5) Lógica de actualización (UI) ---
def update_dashboard(brand_filter, sentiment_filter):
    """
    Filtra el DataFrame en memoria y actualiza los componentes de la UI.
    Devuelve: fig_intel, fig_amd, fig_compare, tabla
    """
    if df_automl_predictions.empty:
        empty_df = pd.DataFrame(columns=["Marca", "Comentario", "Sentimiento (AutoML)"])
        return None, None, None, empty_df

    filtered = df_automl_predictions.copy()

    # Aplicar filtros
    if brand_filter != "Ambas":
        filtered = filtered[filtered["brand"] == brand_filter]
    if sentiment_filter != "Todos":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]

    # Figuras
    if brand_filter == "Ambas":
        intel_df = filtered[filtered["brand"] == "Intel"]
        amd_df = filtered[filtered["brand"] == "AMD"]
        fig_intel = create_sentiment_pie(intel_df, "Intel")
        fig_amd = create_sentiment_pie(amd_df, "AMD")
        fig_compare = create_compare_bar(filtered)
    else:
        brand_df = filtered[filtered["brand"] == brand_filter]
        fig_brand = create_sentiment_pie(brand_df, brand_filter)
        # Para consistencia visual, repetimos el mismo gráfico en ambos espacios
        fig_intel, fig_amd = fig_brand, fig_brand
        # Placeholder comparativo cuando no aplica
        fig_compare = px.bar(title="Comparación Intel vs AMD (disponible cuando 'Ambas')")

    # Tabla (muestra primeras 100 filas)
    table_df = (
        filtered[["brand", "text", "sentiment"]]
        .rename(columns={"brand": "Marca", "text": "Comentario", "sentiment": "Sentimiento (AutoML)"})
        .head(100)
    )
    return fig_intel, fig_amd, fig_compare, table_df

# --- 6) Interfaz (Gradio) ---
custom_theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
).set(
    body_background_fill="#f0f2f5",
    block_background_fill="white",
    block_border_width="1px",
    block_shadow="*shadow_drop_lg",
    button_primary_background_fill="#00a3c0",
    button_primary_background_fill_hover="#00839c",
    button_primary_text_color="white",
)

custom_css = """
h1 { color: #2c3e50; text-align: center; font-weight: 700; }
h3 { color: #333d4d; font-weight: 600; }
p  { color: #5d6776; text-align: center; }
"""

with gr.Blocks(theme=custom_theme, css=custom_css, title="Dashboard de AutoML: Intel vs AMD") as demo:
    gr.Markdown("<h1>📊 Dashboard de Resultados del Pipeline de AutoML</h1>")
    gr.Markdown(
        "<p>Análisis de sentimiento utilizando nuestro modelo propio, entrenado y desplegado automáticamente con un pipeline de MLOps.</p>"
    )

    with gr.Row():
        brand_filter = gr.Dropdown(
            choices=["Ambas", "Intel", "AMD"],
            value="Ambas",
            label="Filtrar por Marca"
        )
        # Fijamos las opciones para evitar etiquetas ruidosas o variantes
        sentiment_filter = gr.Dropdown(
            choices=["Todos", "Positive", "Neutral", "Negative"],
            value="Todos",
            label="Filtrar por Sentimiento"
        )

    with gr.Row():
        intel_plot = gr.Plot(label="Intel")
        amd_plot = gr.Plot(label="AMD")

    with gr.Row():
        compare_plot = gr.Plot(label="Comparación Intel vs AMD")

    gr.Markdown("<h3>Vista Previa de Comentarios</h3>")
    comments_table = gr.DataFrame(
        value=pd.DataFrame(columns=["Marca", "Comentario", "Sentimiento (AutoML)"]),
        wrap=True,
        interactive=False
    )

    gr.Markdown("<h3>Detalles del Modelo AutoML</h3>")
    gr.HTML(value=automl_model_info_str)

    # Conectar
    inputs = [brand_filter, sentiment_filter]
    outputs = [intel_plot, amd_plot, compare_plot, comments_table]
    demo.load(fn=update_dashboard, inputs=inputs, outputs=outputs)
    for w in inputs:
        w.change(fn=update_dashboard, inputs=inputs, outputs=outputs)

# --- 7) Lanzamiento ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
