import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model
import shap
import matplotlib.pyplot as plt
import io
import base64

# Load the pipeline and extract the model
pipeline = load_model('churn_model')
model = pipeline.steps[-1][1]

# Initialize SHAP Explainer
explainer = shap.TreeExplainer(model)

# Prediction Function
def predict(gender, age, tenure, usage_frequency, support_calls, payment_delay,
            subscription_type, contract_length, total_spend, last_interaction):
    
    # Define the schema and create the DataFrame
    input_data = {
        'Gender': [gender], 'Age': [age], 'Tenure': [tenure], 
        'Usage Frequency': [usage_frequency], 'Support Calls': [support_calls],
        'Payment Delay': [payment_delay], 'Subscription Type': [subscription_type],
        'Contract Length': [contract_length], 'Total Spend': [total_spend],
        'Last Interaction': [last_interaction]
    }
    schema = {
        'Gender': 'object', 'Age': 'int64', 'Tenure': 'int64',
        'Usage Frequency': 'int64', 'Support Calls': 'int64',
        'Payment Delay': 'int64', 'Subscription Type': 'object',
        'Contract Length': 'object', 'Total Spend': 'float64',
        'Last Interaction': 'int64'
    }
    input_df = pd.DataFrame(input_data).astype(schema)

    # Use the pipeline to get predictions with raw scores
    predictions = predict_model(pipeline, data=input_df, raw_score=True)
    
    # Select the probability for the 'Churn' class (class 1)
    churn_probability = predictions.loc[0, 'prediction_score_1']
    
    if churn_probability >= 0.5:
        prediction_text = f"<p style='color:red; font-size:1.5em; font-weight:bold;'>¡Alto Riesgo de Fuga ({churn_probability:.2%})!</p>"
    else:
        prediction_text = f"<p style='color:green; font-size:1.5em; font-weight:bold;'>Bajo Riesgo de Fuga ({churn_probability:.2%})</p>"
    
    confianza = f"<p style='text-align:center; font-size:1.1em;'>Probabilidad calculada por el modelo.</p>"

    # SHAP Graph Generation
    try:
        transformed_df = pipeline[:-1].transform(input_df)
        shap_values = explainer.shap_values(transformed_df)
        
        if isinstance(shap_values, list):
            shap_values_for_plot = shap_values[1][0]
            base_value_for_plot = explainer.expected_value[1]
        else:
            shap_values_for_plot = shap_values[0]
            base_value_for_plot = explainer.expected_value

        plt.figure(figsize=(2, 2))


        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_for_plot,
                base_values=base_value_for_plot,
                data=transformed_df.iloc[0],
                feature_names=transformed_df.columns.tolist()
            ),
            show=False
        )
        
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        shap_plot = f"<img src='data:image/png;base64,{image_base64}' style='width:100%; height:auto;'/>"

    except Exception as e:
        shap_plot = f"<p style='color:red;'>Error al generar gráfico SHAP: {e}</p>"
        print(f"SHAP Error: {e}")
        
    return prediction_text, confianza, shap_plot

# --- Custom CSS and Gradio UI Layout ---
custom_css = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; }
.gradio-container { box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 12px; overflow: hidden; }
h1 { color: #2c3e50; text-align: center; font-weight: bold; margin-bottom: 20px; }
p { color: #555; text-align: center; font-size: 1.1em; }
.gr-button.primary { background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-size: 1.1em; transition: background-color 0.3s ease; }
.gr-button.primary:hover { background-color: #2980b9; }
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Predicción de Fuga de Clientes", css=custom_css) as demo:
    gr.Markdown(
        """
        <h1>📈 Predicción de Fuga de Clientes (Churn)</h1>
        <p>Introduce los datos de un cliente para predecir su riesgo de cancelar la suscripción y entender los factores clave detrás de la predicción.</p>
        """
    )
    
    # --- CAMBIO: Reorganización en dos columnas principales ---
    with gr.Row():
        # --- Columna Izquierda: Entradas del Usuario ---
        with gr.Column(scale=1):
            gr.Markdown("<h3>Datos del Cliente</h3>")
            gender = gr.Radio(["Male", "Female"], label="Género")
            age = gr.Slider(minimum=18, maximum=70, step=1, label="Edad")
            tenure = gr.Slider(minimum=0, maximum=60, step=1, label="Antigüedad (meses)")
            usage_frequency = gr.Slider(minimum=1, maximum=30, step=1, label="Frecuencia de Uso (días/mes)")
            support_calls = gr.Slider(minimum=0, maximum=10, step=1, label="Llamadas a Soporte")
            payment_delay = gr.Slider(minimum=0, maximum=30, step=1, label="Retraso en Pagos (días)")
            subscription_type = gr.Dropdown(["Basic", "Standard", "Premium"], label="Tipo de Suscripción")
            contract_length = gr.Dropdown(["Monthly", "Quarterly", "Annual"], label="Duración del Contrato")
            total_spend = gr.Number(label="Gasto Total ($)")
            last_interaction = gr.Slider(minimum=1, maximum=30, step=1, label="Última Interacción (días)")
            
            predict_btn = gr.Button("Predecir Churn", variant="primary")

        # --- Columna Derecha: Salidas (Resultado y SHAP) ---
        with gr.Column(scale=2):
            gr.Markdown("<h3>Resultado y Explicación del Modelo</h3>")
            output_label = gr.HTML(label="Predicción")
            output_score = gr.HTML(label="Detalle")
            shap_output = gr.HTML(label="Factores Clave (SHAP)")

    # Conectar el botón con la función y las I/O
    predict_btn.click(
        fn=predict,
        inputs=[gender, age, tenure, usage_frequency, support_calls, payment_delay,
                subscription_type, contract_length, total_spend, last_interaction],
        outputs=[output_label, output_score, shap_output]
    )
    
    # Ejemplos para probar la app
    gr.Examples(
        examples=[
            ["Male", 30, 12, 15, 1, 0, "Basic", "Monthly", 250.0, 7],
            ["Female", 45, 36, 20, 3, 5, "Premium", "Annual", 1500.0, 15],
            ["Male", 55, 6, 5, 5, 20, "Standard", "Monthly", 100.0, 25]
        ],
        inputs=[gender, age, tenure, usage_frequency, support_calls, payment_delay,
                subscription_type, contract_length, total_spend, last_interaction],
    )

demo.launch()