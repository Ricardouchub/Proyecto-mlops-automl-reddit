import os
import re
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
    if not p:
        return DEFAULT_RESULTS_PATH
    p = p.replace("\r", "").replace("\n", "")
    if p.startswith(REPO_FOLDER_NAME + os.sep):
        p = p[len(REPO_FOLDER_NAME + os.sep):]
    return os.path.normpath(p)

def _read_csv_safe(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8")

# --- Mapeo agresivo de marcas ---
INTEL_PAT = re.compile(r"\bintel\b", flags=re.IGNORECASE)
AMD_PAT   = re.compile(r"\bamd\b",   flags=re.IGNORECASE)

def _map_brand_any(value: str) -> str:
    s = str(value).strip()
    if INTEL_PAT.search(s):
        return "Intel"
    if AMD_PAT.search(s):
        return "AMD"
    return s.strip().title()

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    if "brand" in df.columns:
        df["brand_original"] = df["brand"].astype(str)  # para diagnóstico
        df["brand"] = df["brand"].apply(_map_brand_any)

    if "sentiment" in df.columns:
        df["sentiment"] = (
            df["sentiment"]
            .astype(str).str.strip().str.title()
            .replace({
                "Positivo": "Positive",
                "Negativo": "Negative",
                "Neutralidad": "Neutral",
                "Neutro": "Neutral",
            })
        )
        df = df[df["sentiment"].isin(SENTIMENT_ORDER)]

    return df

def _prep_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve DataFrame con ['sentiment','count'] para alimentar el pie.
    Fuerza 'count' a numérico (int) y ordena por SENTIMENT_ORDER.
    """
    if df.empty or "sentiment" not in df.columns:
        return pd.DataFrame(columns=["sentiment", "count"])
    counts = (df["sentiment"].value_counts()
              .rename_axis("sentiment").reset_index(name="count"))
    # orden + tipos
    counts = counts[counts["sentiment"].isin(SENTIMENT_ORDER)]
    counts["count"] = pd.to_numeric(counts["count"], errors="coerce").fillna(0).astype(int)
    # reordena según SENTIMENT_ORDER
    order_index = {s:i for i,s in enumerate(SENTIMENT_ORDER)}
    counts = counts.sort_values(key=lambda s: s.map(order_index) if s.name=="sentiment" else s)
    return counts

def _prep_brand_sentiment_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"brand", "sentiment"}.issubset(df.columns):
        return pd.DataFrame(columns=["brand", "sentiment", "count"])
    g = (df.groupby(["brand", "sentiment"]).size().reset_index(name="count"))
    g = g[g["sentiment"].isin(SENTIMENT_ORDER)]
    g["count"] = pd.to_numeric(g["count"], errors="coerce").fillna(0).astype(int)
    # reordena
    s_idx = {s:i for i,s in enumerate(SENTIMENT_ORDER)}
    b_idx = {b:i for i,b in enumerate(BRAND_ORDER)}
    g = g.sort_values(by=["sentiment","brand"], key=lambda col:
                      col.map(s_idx) if col.name=="sentiment" else col.map(b_idx))
    return g

# ---------- AQUI EL CAMBIO CLAVE ----------
def create_sentiment_pie(data: pd.DataFrame, brand_name: str):
    """
    Construye el pie pasando listas explícitas (names/values) para evitar que
    Plotly/serialización ignore la columna 'count'.
    """
    counts = _prep_counts(data)
    if counts.empty:
        return px.pie(title=f"Sin Datos de Sentimiento para {brand_name}")

    names_list = counts["sentiment"].astype(str).tolist()
    values_list = counts["count"].astype(float).tolist()  # float por seguridad en versiones antiguas

    fig = px.pie(
        names=names_list,
        values=values_list,
        title=f"Sentimiento para {brand_name} (Modelo AutoML)",
        color=names_list,  # asegura el mapping por nombre
        color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"},
    )
    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value} (%{percent})")

    # DEBUG: imprime los valores que realmente está usando el trace
    try:
        print(f"[DEBUG] PIE {brand_name} values:", fig.data[0]["values"])
        print(f"[DEBUG] PIE {brand_name} labels:", fig.data[0]["labels"])
    except Exception as _e:
        print(f"[DEBUG] No se pudo leer values/labels del pie ({brand_name}):", _e)

    return fig
# ------------------------------------------

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
env_path = os.environ.get("AUTOML_RESULTS_PATH", DEFAULT_RESULTS_PATH)
AUTOML_RESULTS_PATH = _normalize_input_path(env_path)

df_automl_predictions = pd.DataFrame()
automl_model_info_str = (
    "La información del modelo no se encontró. Asegúrate de que el pipeline de entrenamiento se haya ejecutado."
)

def load_data():
    global df_automl_predictions
    try:
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

        print("[DEBUG] shape post-normalization:", df.shape)
        if set(["brand", "sentiment"]).issubset(df.columns):
            print("[DEBUG] brand x sentiment:\n",
                  df.groupby(["brand", "sentiment"]).size().reset_index(name="count").to_string(index=False))
            if "brand_original" in df.columns:
                print("[DEBUG] Top brands originales:\n",
                      df["brand_original"].str.strip().value_counts().head(10).to_string())

        df_automl_predictions = df
        return True, f"OK: {abs_path}", df
    except Exception as e:
        print(f"Error al cargar '{AUTOML_RESULTS_PATH}': {e}")
        df_automl_predictions = pd.DataFrame(columns=["brand", "text", "sentiment"])
        return False, str(e), df_automl_predictions

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
    """
    Devuelve:
      fig_intel, fig_amd, fig_compare, table_df,
      intel_counts_tbl, amd_counts_tbl
    """
    if df_automl_predictions.empty:
        empty_df = pd.DataFrame(columns=["Marca", "Comentario", "Sentimiento (AutoML)"])
        empty_counts = pd.DataFrame(columns=["sentiment", "count"])
        return None, None, None, empty_df, empty_counts, empty_counts

    filtered = df_automl_predictions.copy()
    if brand_filter != "Ambas":
        filtered = filtered[filtered["brand"] == brand_filter]
    if sentiment_filter != "Todos":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]

    if brand_filter == "Ambas":
        intel_df = filtered[filtered["brand"] == "Intel"]
        amd_df   = filtered[filtered["brand"] == "AMD"]

        intel_counts = _prep_counts(intel_df)
        amd_counts   = _prep_counts(amd_df)

        fig_intel = create_sentiment_pie(intel_df, "Intel")
        fig_amd   = create_sentiment_pie(amd_df, "AMD")
        fig_compare = create_compare_bar(filtered)
    else:
        brand_df     = filtered[filtered["brand"] == brand_filter]
        brand_counts = _prep_counts(brand_df)
        fig_brand    = create_sentiment_pie(brand_df, brand_filter)
        fig_intel, fig_amd = fig_brand, fig_brand
        fig_compare  = px.bar(title="Comparación Intel vs AMD (disponible cuando 'Ambas')")

        intel_counts = brand_counts
        amd_counts   = brand_counts

    table_df = (
        filtered[["brand", "text", "sentiment"]]
        .rename(columns={"brand": "Marca", "text": "Comentario", "sentiment": "Sentimiento (AutoML)"})
        .head(100)
    )
    return fig_intel, fig_amd, fig_compare, table_df, intel_counts, amd_counts

def get_diagnostics():
    path = os.path.abspath(AUTOML_RESULTS_PATH)
    exists = os.path.exists(AUTOML_RESULTS_PATH)
    size = os.path.getsize(AUTOML_RESULTS_PATH) if exists else 0
    shape = df_automl_predictions.shape
    g = _prep_brand_sentiment_counts(df_automl_predictions)
    topb = pd.DataFrame()
    if "brand_original" in df_automl_predictions.columns:
        topb = (
            df_automl_predictions["brand_original"]
            .astype(str).str.strip()
            .value_counts()
            .head(12)
            .reset_index()
            .rename(columns={"index": "brand_original", "brand_original": "count"})
        )
    return f"Ruta efectiva: {path}", f"Existe: {exists}\nTamaño(bytes): {size}\nShape: {shape}", topb, g

def reload_csv():
    ok, msg, _df = load_data()
    diag_path, diag_info, topb, g = get_diagnostics()
    return msg, diag_path, diag_info, topb, g

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
        amd_plot   = gr.Plot(label="AMD")

    with gr.Row():
        intel_counts_tbl = gr.DataFrame(label="Conteos (Intel) usados por el gráfico")
        amd_counts_tbl   = gr.DataFrame(label="Conteos (AMD) usados por el gráfico")

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
        diag_status = gr.Textbox(label="Estado de carga", interactive=False)
        diag_path = gr.Textbox(label="Ruta", interactive=False)
        diag_info = gr.Textbox(label="Resumen (exists/size/shape)", lines=3, interactive=False)
        diag_topbrands = gr.DataFrame(label="Top marcas originales (sin mapear) - Para verificar")
        diag_counts = gr.DataFrame(label="brand × sentiment (conteos)")

        reload_btn = gr.Button("Reload CSV")
        reload_btn.click(fn=reload_csv, inputs=None, outputs=[diag_status, diag_path, diag_info, diag_topbrands, diag_counts])

    gr.Markdown("<h3>Detalles del Modelo AutoML</h3>")
    try:
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
            automl_model_info_str = f.read()
    except Exception:
        pass
    gr.HTML(value=automl_model_info_str)

    # Conectar
    inputs = [brand_filter, sentiment_filter]
    outputs = [intel_plot, amd_plot, compare_plot, comments_table, intel_counts_tbl, amd_counts_tbl]
    demo.load(fn=update_dashboard, inputs=inputs, outputs=outputs)
    for w in inputs:
        w.change(fn=update_dashboard, inputs=inputs, outputs=outputs)

    # Cargar diagnóstico al inicio
    def _init_diag():
        ok, msg, _df = load_data()
        return (msg, ) + get_diagnostics()
    demo.load(fn=_init_diag, inputs=None, outputs=[diag_status, diag_path, diag_info, diag_topbrands, diag_counts])

# --------------------------
# 5) Launch
# --------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)