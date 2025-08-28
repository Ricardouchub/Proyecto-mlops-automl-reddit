# 1. Importar las librerías necesarias
import pandas as pd
from pycaret.regression import *

# 2. Cargar el conjunto de datos
print("Cargando datos...")
dataset = pd.read_csv('insurance.csv')
print("Datos cargados exitosamente.")

# 3. Configurar el entorno de PyCaret (¡AQUÍ ESTÁ LA MAGIA!)
print("Configurando el experimento de regresión con MLflow...")
s = setup(data=dataset, 
          target='charges', 
          session_id=123,
          log_experiment=True,                       # <-- CAMBIO: Activa el registro
          experiment_name='insurance_cost_prediction', # <-- CAMBIO: Nombra el experimento
          verbose=False)
print("Configuración completa.")

# 4. Comparar todos los modelos
print("Buscando el mejor modelo...")
best_model = compare_models()
print("Mejor modelo encontrado:")
print(best_model)

# 5. Finalizar y guardar el modelo
# Finalize_model re-entrena el modelo con todos los datos
print("Finalizando y guardando el modelo...")
final_model = finalize_model(best_model)
save_model(final_model, 'best_insurance_model')
print("¡Modelo guardado! Proceso completado.")