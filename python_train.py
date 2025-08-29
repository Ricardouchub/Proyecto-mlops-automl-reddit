import pandas as pd
from pycaret.classification import *

# Carga  dataset
print("Cargando datos de Churn...")
dataset = pd.read_csv('churn_data.csv')
# Eliminamos la columna de Customer ID ya que no es útil para la predicción
dataset = dataset.drop('CustomerID', axis=1)
print("Datos cargados exitosamente.")

# Configurar el experimento de clasificación
print("Configurando el experimento de clasificación con MLflow...")
s = setup(data=dataset, 
          target='Churn', 
          session_id=123,
          log_experiment=True,
          experiment_name='churn_prediction', 
          verbose=False)
print("Configuración completa.")

# Comparar modelos y encontrar el mejor
print("Buscando el mejor modelo de clasificación...")
best_model = compare_models()
print("Mejor modelo encontrado:")
print(best_model)

# Finalizar y guardar el modelo
print("Finalizando y guardando el modelo...")
final_model = finalize_model(best_model)
save_model(final_model, 'churn_model')
print("¡Modelo guardado! Proceso completado.")