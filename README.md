# Pipeline de MLOps para Análisis de Sentimiento en Tiempo Real: Intel vs. AMD en Reddit

Este proyecto demuestra la construcción de un **pipeline de MLOps de extremo a extremo**, diseñado para ser completamente autónomo. El sistema extrae datos en vivo desde la API de Reddit, los procesa utilizando múltiples modelos de NLP, entrena un modelo de Machine Learning propio con AutoML y presenta los resultados en un dashboard interactivo.

## Características Principales

* **Extracción de Datos Automatizada:** Un bot se conecta diariamente a la API de Reddit para buscar y recolectar nuevos comentarios sobre Intel y AMD en subreddits de tecnología.
* **Procesamiento Avanzado de NLP:** Cada comentario es enriquecido con un análisis multifacético:
    * **Sentimiento:** Positivo, Negativo o Neutral.
    * **Emociones:** Alegría, Ira, Miedo, etc.
    * **Modelado de Tópicos:** Descubrimiento de los principales temas de conversación.
* **Entrenamiento con AutoML:** Se entrena un modelo de clasificación de sentimiento desde cero utilizando **PyCaret**, demostrando un pipeline de entrenamiento automatizado.
* **Dashboard Comparativo:** Una interfaz interactiva construida con **Gradio** que permite comparar el rendimiento del modelo propio (AutoML) contra un modelo experto de última generación (RoBERTa), demostrando una toma de decisiones basada en datos.
* **Orquestación con GitHub Actions:** Dos pipelines de CI/CD interconectados que automatizan todo el flujo, desde la extracción de datos hasta el re-entrenamiento del modelo.

## ⚙️ El Pipeline de MLOps en Acción

Este proyecto no es solo un modelo, es un **sistema vivo y autónomo**. La magia reside en la orquestación de dos bots (workflows de GitHub Actions) que trabajan en conjunto:

#### ** Bot 1: El Extractor de Datos (Diario)**

* **Activación:** Se ejecuta automáticamente todos los días a las 05:00 UTC.
* **Misión:**
    1.  `▶️` Ejecuta el script `extract_reddit_data.py`.
    2.  `📡` Se conecta a la API de Reddit y busca nuevos comentarios sobre Intel y AMD.
    3.  `💾` Actualiza el archivo `data/reddit_comments.csv` con los nuevos hallazgos.
    4.  `⬆️` Hace `commit` y `push` del archivo actualizado de vuelta al repositorio.

#### ** Bot 2: El Procesador y Entrenador (Reactivo)**

* **Activación:** Se dispara **inmediatamente después** de que el Bot 1 sube los nuevos datos.
* **Misión:**
    1.  `▶️` Ejecuta el script `nlp_processor.py`.
    2.  `🧠` Toma los comentarios crudos y los enriquece con análisis de sentimiento, emociones y tópicos, guardando el resultado en `data/processed_reddit_data.csv`.
    3.  `▶️` Ejecuta el script `python_train.py`.
    4.  `🏋️‍♂️` Entrena un nuevo modelo de sentimiento desde cero utilizando **AutoML (PyCaret)**.
    5.  `🎨` Guarda los artefactos del entrenamiento: el modelo (`models/sentiment_model_v2.pkl`) y el gráfico de importancia de características.
    6.  `⬆️` Hace `commit` y `push` de todos los artefactos actualizados al repositorio.

Este ciclo asegura que el proyecto se mantenga siempre al día con la conversación más reciente, mejorando y adaptándose sin intervención manual.

##  La Historia: AutoML vs. Modelo Experto

Uno de los hallazgos clave de este proyecto es la comparación directa entre un modelo entrenado por nosotros y un modelo pre-entrenado de última generación (State-of-the-Art).

* **Nuestro Modelo (AutoML):** Como se puede ver en el dashboard, el modelo entrenado con PyCaret, a pesar de los esfuerzos, obtiene un rendimiento bajo. Esto demuestra empíricamente que el lenguaje de Reddit es demasiado complejo y lleno de matices para los modelos de ML tradicionales basados en frecuencia de palabras.
   * **Resultados del Mejor Modelo (Ridge Classifier):**
   * **Accuracy:** `0.5489` (apenas por encima del azar para 3 clases).
   * **AUC:** `0.0000` (incapaz de discriminar entre clases).
   * **Kappa:** `0.1729` (rendimiento muy bajo por encima del azar).
* **Modelo Experto (RoBERTa):** Para la aplicación final, se toma la decisión estratégica de usar un modelo Transformer pre-entrenado (`cardiffnlp/twitter-roberta-base-sentiment-latest`), que ofrece una precisión muy superior al entender el contexto del lenguaje.

Esta comparación no es un fallo, sino la **demostración de un proceso de MLOps maduro**: saber experimentar, analizar resultados y elegir la mejor herramienta para el producto final.


##  Estructura del Proyecto


<img width="590" height="434" alt="image" src="https://github.com/user-attachments/assets/d351cd08-88ba-4c4d-8c70-4c53c27d03f6" />
