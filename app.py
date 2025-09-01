import os
import pandas as pd
import plotly.express as px
import gradio as gr

# --------------------------
# 1) Config y utilidades
# --------------------------
REPO_FOLDER_NAME = "proyecto-mlops-reddit"  # por si la ruta viene con prefijo
DEFAULT_RESULTS_PATH = os.path.join("data", "automl_results.csv")
MODELS_FOLDER = "models"
MODEL_INFO_PATH = os.path.join(MODELS_FOLDER, "model_info.txt")

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
BRAND_ORDER = ["Intel", "AMD"]

def _normalize_input_path(p: str) -> str:
    """
    Recibe una ruta que podría venir con prefijo 'proyecto-mlops-reddit/' y lo elimina.
    También limpia caracteres de salto de línea o retorno de carro (\r, \n).
    """
    if not p:
        return DEFAULT_RESULTS_PATH
    p = p.replace("\r", "").replace("\n", "")
    # Quita prefijo del repo si viene incluido
    if p.startswith(REPO_FOLDER_NAME + os.sep):
        p = p[len(REPO_FOLDER_NAME + os.sep):]
    # Normaliza separadores
    return os.path.normpath(p)

def _read_csv_safe(path: str) -> pd.DataFrame:
    """Lee un CSV intentando con distintas codificaciones."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8")

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas clave para evitar variaciones de etiqueta."""
    if df.empty:
        return df
    df = df.copy()

    if "brand" in df.columns:
        df["brand"] = (
            df["brand"].astype(str).str.strip().str.title().replace({"Amd": "AMD"})
        )
    if "sentiment" in df.columns:
        df["sentiment"] = (
            df["sentiment"].astype(str).str.strip().str.title().replace({
                "Positivo": "Positive",
                "Negativo": "Negative",
                "Neutralidad": "Neutral",
                "Neutro": "Neutral",
            })
        )
        df = df[df["sentiment"].isin(SENTIMENT_ORDER)]
    return df

def _prep_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["sentiment", "count"])
    counts = (df["sentiment"].value_counts()
              .rename_axis("sentiment").reset_index(name="count"))
    counts = counts[counts["sentiment"].isin(SENTIMENT_ORDER)]
    counts["sentiment"] = pd.Categorical(counts["sentiment"], SENTIMENT_ORDER, True)
    return counts.sort_values("sentiment")

def _prep_brand_sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"brand", "sentiment"}.issubset(df.columns):
        return pd.DataFrame(columns=["brand", "sentiment", "count"])
    g = (df.groupby(["brand", "sentiment"]).size().reset_index(name="count"))
    g = g[g["sentiment"].isin(SENTIMENT_ORDER)]
    g["sentiment"] = pd.Categorical(g["sentiment"], SENTIMENT_ORDER, True)
    g["brand"] = pd.Categorical(g["brand"], BRAND_ORDER, True)
    return g.sort_values(["sentiment", "brand"])

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
    g = _prep_brand_sentiment_counts(data)
    if g.empty:
        return px.bar(title="Comparación Intel vs AMD (sin datos)")
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

# --------------------------
# 2) Carga inicial
# --------------------------
# Permite override por env var (útil en Render)
env_path = os.environ.get("AUTOML_RESULTS_PATH", DEFAULT_RESULTS_PATH)
AUTOML_RESULTS_PATH = _normalize_input_path(env_path)

df_automl_predictions = pd.DataFrame()
automl_model_info_str = (
    "La información del modelo no se encontró. Asegúrate de que el pipeline de entrenamiento se haya ejecutado."
)

def load_data():
    global df_automl_predictions
    try:
        # DEBUG a logs
        abs_path = os.path.abspath(AUTOML_RESULTS_PATH)
        print("[DEBUG] LEYENDO CSV:", abs_path)
        print("[DEBUG] exists?:", os.path.exists(AUTOML_RESULTS_PATH))
        if os.path.exists(AUTOML_RESULTS_PATH):
            try:
                print("[DEBUG] size(bytes):", os.path.getsize(AUTOML_RESULTS_PATH))
            except Exception as _e:
                print("[DEBUG] os.path.getsize error:", _e)
        df = _read_csv_safe(AUTOML_RESULTS_PATH)
        df = _normalize_df(df)
        print("[DEBUG] shape:", df.shape)
        if set(["brand", "sentiment"]).issubset(df.columns):
            print("[DEBUG] brand x sentiment:\n",
                  df.groupby(["brand", "sentiment"]).size().reset_index(name="count").to_string(index=False))
        df_automl_predictions = df
        return True, f"OK: {abs_path}", df
    except Exception as e:
        print(f"Error al cargar '{AUTOML_RESULTS_PATH}': {e}")
        df_automl_predictions = pd.DataFrame(columns=["brand", "text", "sentiment"])
        return False, str(e), df_automl_predictions

# Carga inicial
_ = load_data()

try:
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        automl_model_info_str = f.read()
except Exception as e:
    print(f"Advertencia: No se pudo cargar '{MODEL_INFO_PATH}': {e}")

# --------------------------
# 3) Callbacks UI
# --------------------------
def update_dashboard(brand_filter, sentiment_filter):
    if df_automl_predictions.empty:
        empty_df = pd.DataFrame(columns=["Marca", "Comentario", "Sentimiento (AutoML)"])
        return None, None, None, empty_df

    filtered = df_automl_predictions.copy()
    if brand_filter != "Ambas":
        filtered = filtered[filtered["brand"] == brand_filter]
    if sentiment_filter != "Todos":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]

    if brand_filter == "Ambas":
        intel_df = filtered[filtered["brand"] == "Intel"]
        amd_df = filtered[filtered["brand"] == "AMD"]
        fig_intel = create_sentiment_pie(intel_df, "Intel")
        fig_amd = create_sentiment_pie(amd_df, "AMD")
        fig_compare = create_compare_bar(filtered)
    else:
        brand_df = filtered[filtered["brand"] == brand_filter]
        fig_brand = create_sentiment_pie(brand_df, brand_filter)
        fig_intel, fig_amd = fig_brand, fig_brand
        fig_compare = px.bar(title="Comparación Intel vs AMD (disponible cuando 'Ambas')")

    table_df = (
        filtered[["brand", "text", "sentiment"]]
        .rename(columns={"brand": "Marca", "text": "Comentario", "sentiment": "Sentimiento (AutoML)"})
        .head(100)
    )
    return fig_intel, fig_amd, fig_compare, table_df

def get_diagnostics():
    """Devuelve info de diagnóstico visible en UI."""
    path = os.path.abspath(AUTOML_RESULTS_PATH)
    exists = os.path.exists(AUTOML_RESULTS_PATH)
    size = os.path.getsize(AUTOML_RESULTS_PATH) if exists else 0
    shape = df_automl_predictions.shape
    g = _prep_brand_sentiment_counts(df_automl_predictions)
    return f"Ruta efectiva: {path}\nExiste: {exists}\nTamaño(bytes): {size}\nShape: {shape}", g

def reload_csv():
    ok, msg, _df = load_data()
    diag, g = get_diagnostics()
    return msg, diag, g

# --------------------------
# 4) UI (Gradio)
# --------------------------
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
pre { background:#f7f7f7; padding:12px; border-radius:8px; }
"""

with gr.Blocks(theme=custom_theme, css=custom_css, title="Resultados del Pipeline de AutoML") as demo:
    gr.Markdown("<h1>📊 Resultados del Pipeline de AutoML</h1>")
    gr.Markdown("<p>Análisis de sentimiento utilizando nuestro modelo propio, entrenado y desplegado automáticamente con un pipeline de MLOps.</p>")

    with gr.Row():
        brand_filter = gr.Dropdown(choices=["Ambas", "Intel", "AMD"], value="Ambas", label="Filtrar por Marca")
        sentiment_filter = gr.Dropdown(choices=["Todos", "Positive", "Neutral", "Negative"], value="Todos", label="Filtrar por Sentimiento")

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

    # --- Panel de Diagnóstico ---
    with gr.Accordion("🔎 Diagnóstico de datos (para verificar CSV en Render)", open=False):
        diag_path = gr.Textbox(label="Estado de carga", interactive=False)
        diag_info = gr.Textbox(label="Resumen (ruta/exists/size/shape)", lines=4, interactive=False)
        diag_table = gr.DataFrame(label="brand × sentiment (conteos)")

        reload_btn = gr.Button("Reload CSV")
        reload_btn.click(fn=reload_csv, inputs=None, outputs=[diag_path, diag_info, diag_table])

    gr.Markdown("<h3>Detalles del Modelo AutoML</h3>")
    try:
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
            automl_model_info_str = f.read()
    except Exception:
        pass
    gr.HTML(value=automl_model_info_str)

    # Conectar
    inputs = [brand_filter, sentiment_filter]
    outputs = [intel_plot, amd_plot, compare_plot, comments_table]
    demo.load(fn=update_dashboard, inputs=inputs, outputs=outputs)

    # Cargar diagnóstico al inicio
    def _init_diag():
        ok, msg, _df = load_data()
        diag, g = get_diagnostics()
        return msg, diag, g
    demo.load(fn=_init_diag, inputs=None, outputs=[diag_path, diag_info, diag_table])

    for w in inputs:
        w.change(fn=update_dashboard, inputs=inputs, outputs=outputs)

# --------------------------
# 5) Launch
# --------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)