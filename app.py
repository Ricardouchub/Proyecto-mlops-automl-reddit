import gradio as gr
import pandas as pd
from pycaret.regression import load_model, predict_model

# Cargar el modelo entrenado
model = load_model('best_insurance_model')

# Definir la función de predicción
def predict(age, sex, bmi, children, smoker, region):
    # Crear un DataFrame de pandas con los inputs del usuario
    data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    }
    input_df = pd.DataFrame(data)

    # Usar el modelo para hacer la predicción
    predictions = predict_model(model, data=input_df)

    # Extraer el valor de la predicción
    # La columna se llama 'prediction_label' en PyCaret 3.x
    costo_predicho = predictions.loc[0, 'prediction_label']

    # Formatear el resultado como moneda
    return f"${costo_predicho:,.2f} USD"

# --- Creación de la Interfaz con Gradio ---

# Usaremos gr.Blocks para un mayor control del diseño
with gr.Blocks(theme=gr.themes.Soft(), title="Estimador de Costos de Seguro") as demo:
    # Título y descripción
    gr.Markdown(
        """
        # Estimador de Costos de Seguros Médicos 🏥
        Introduce tus datos para obtener una estimación del costo anual de tu seguro médico.
        Este modelo fue entrenado usando AutoML con PyCaret.
        """
    )

    with gr.Row():
        with gr.Column():
            # --- Componentes de Entrada ---
            age = gr.Slider(minimum=18, maximum=100, step=1, label="Edad")
            sex = gr.Radio(["male", "female"], label="Sexo")
            bmi = gr.Number(label="Índice de Masa Corporal (IMC)")

        with gr.Column():
            children = gr.Slider(minimum=0, maximum=10, step=1, label="Número de Hijos")
            smoker = gr.Radio(["yes", "no"], label="¿Es Fumador?")
            region = gr.Dropdown(
                ["southwest", "southeast", "northwest", "northeast"], 
                label="Región"
            )

    # Botón para ejecutar la predicción
    predict_btn = gr.Button("Estimar Costo", variant="primary")

    # --- Componente de Salida ---
    output_cost = gr.Textbox(label="Costo Estimado del Seguro")

    # Conectar el botón con la función y las I/O
    predict_btn.click(
        fn=predict,
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=output_cost
    )

    # Ejemplos para que el usuario pueda probar fácilmente
    gr.Examples(
        examples=[
            [25, "male", 22.5, 0, "no", "northwest"],
            [45, "female", 33.1, 2, "yes", "southeast"],
        ],
        inputs=[age, sex, bmi, children, smoker, region],
    )

# Lanzar la aplicación
demo.launch()