import os
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

# --- Config mínima ---
AUTOML_RESULTS_PATH = "data/automl_results.csv"
MODEL_INFO_PATH     = "models/model_info.txt"
SENTIMENTS          = ["Positive", "Neutral", "Negative"]
BRANDS              = ["Intel", "AMD"]
PALETTE             = {"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}

# --- Carga simple del CSV ---
try:
    try:
        df = pd.read_csv(AUTOML_RESULTS_PATH, encoding="utf-8")
    except Exception:
        df = pd.read_csv(AUTOML_RESULTS_PATH, encoding="latin-1")
except Exception as e:
    print(f"[ERROR] No pude leer {AUTOML_RESULTS_PATH}: {e}")
    df = pd.DataFrame(columns=["brand", "text", "sentiment"])

# --- Normalización mínima (solo Intel/AMD y 3 sentimientos) ---
if not df.empty:
    # Marcas: solo Intel/AMD (case-insensitive). Descarto el resto.
    m = {"intel": "Intel", "amd": "AMD"}
    df["brand"] = df["brand"].astype(str).str.strip().str.casefold().map(m)
    df = df[df["brand"].isin(BRANDS)]

    # Sentimientos: acepto equivalentes en español y dejo solo 3 etiquetas finales
    df["sentiment"] = (
        df["sentiment"].astype(str).str.strip().str.lower()
        .replace({
            "positivo": "positive",
            "negativo": "negative",
            "neutralidad": "neutral",
            "neutro": "neutral",
        })
        .map({"positive":"Positive", "neutral":"Neutral", "negative":"Negative"})
    )
    df = df[df["sentiment"].isin(SENTIMENTS)]

# --- Callback único de Gradio ---
def update_dashboard(brand_filter, sentiment_filter):
    if df.empty:
        empty = pd.DataFrame(columns=["Marca", "Comentario", "Sentimiento (AutoML)"])
        return None, None, None, empty

    # Filtrado
    filtered = df.copy()
    if brand_filter != "Ambas":
        filtered = filtered[filtered["brand"] == brand_filter]
    if sentiment_filter != "Todos":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]

    # ---- PIE Intel ----
    intel_sub = filtered[filtered["brand"] == "Intel"]
    intel_counts = intel_sub["sentiment"].value_counts()
    intel_values = [int(intel_counts.get(s, 0)) for s in SENTIMENTS]
    fig_intel = go.Figure(data=[
        go.Pie(
            labels=SENTIMENTS,
            values=intel_values,
            marker=dict(colors=[PALETTE[s] for s in SENTIMENTS]),
            sort=False,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    ])
    fig_intel.update_layout(title="Sentimiento para Intel (Modelo AutoML)")

    # ---- PIE AMD ----
    amd_sub = filtered[filtered["brand"] == "AMD"]
    amd_counts = amd_sub["sentiment"].value_counts()
    amd_values = [int(amd_counts.get(s, 0)) for s in SENTIMENTS]
    fig_amd = go.Figure(data=[
        go.Pie(
            labels=SENTIMENTS,
            values=amd_values,
            marker=dict(colors=[PALETTE[s] for s in SENTIMENTS]),
            sort=False,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    ])
    fig_amd.update_layout(title="Sentimiento para AMD (Modelo AutoML)")

    # ---- Barra comparativa (Intel vs AMD) ----
    g = (
        filtered.groupby(["sentiment", "brand"]).size()
        .unstack("brand", fill_value=0)
        .reindex(index=SENTIMENTS, columns=BRANDS, fill_value=0)
    )
    bar = go.Figure()
    for b in BRANDS:
        bar.add_trace(go.Bar(
            name=b,
            x=g.index.tolist(),
            y=g[b].astype(float).tolist(),
            hovertemplate=b + " – %{x}: %{y:,}<extra></extra>",
        ))
    bar.update_layout(
        barmode="group",
        title="Comparación Intel vs AMD por Sentimiento",
        xaxis_title="Sentimiento",
        yaxis_title="Conteo",
        legend_title_text="brand",
    )
    bar.update_yaxes(tickformat=",d")

    # ---- Tabla de comentarios ----
    table = (
        filtered[["brand", "text", "sentiment"]]
        .rename(columns={"brand":"Marca", "text":"Comentario", "sentiment":"Sentimiento (AutoML)"})
        .head(100)
    )

    return fig_intel, fig_amd, bar, table

# --- UI sencilla ---
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

with gr.Blocks(theme=custom_theme, title="Resultados del Pipeline de AutoML") as demo:
    gr.Markdown("<h1>📊 Resultados del Pipeline de AutoML</h1>")
    gr.Markdown("<p>Análisis de sentimiento utilizando nuestro modelo propio, entrenado y desplegado automáticamente con un pipeline de MLOps.</p>")

    with gr.Row():
        brand_filter = gr.Dropdown(choices=["Ambas", "Intel", "AMD"], value="Ambas", label="Filtrar por Marca")
        sentiment_filter = gr.Dropdown(choices=["Todos", "Positive", "Neutral", "Negative"], value="Todos", label="Filtrar por Sentimiento")

    with gr.Row():
        intel_plot = gr.Plot(label="Intel")
        amd_plot   = gr.Plot(label="AMD")

    compare_plot = gr.Plot(label="Comparación Intel vs AMD")

    gr.Markdown("<h3>Vista Previa de Comentarios</h3>")
    comments_table = gr.DataFrame(wrap=True, interactive=False)

    inputs  = [brand_filter, sentiment_filter]
    outputs = [intel_plot, amd_plot, compare_plot, comments_table]
    demo.load(fn=update_dashboard, inputs=inputs, outputs=outputs)
    for w in inputs:
        w.change(fn=update_dashboard, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)