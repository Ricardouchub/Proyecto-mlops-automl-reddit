# 🚀 Proyecto Futurista: Pipeline de MLOps para Análisis de Sentimiento en Tiempo Real

### Análisis Competitivo Automatizado: Intel vs. AMD en Reddit

Este proyecto demuestra la construcción de un **pipeline de MLOps de extremo a extremo**, diseñado para ser completamente autónomo. El sistema extrae datos en vivo desde la API de Reddit, los procesa utilizando múltiples modelos de NLP, entrena un modelo de Machine Learning propio con AutoML y presenta los resultados en un dashboard interactivo.

## 📋 Características Principales

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

#### **🤖 Bot 1: El Extractor de Datos (Diario)**

* **Activación:** Se ejecuta automáticamente todos los días a las 05:00 UTC.
* **Misión:**
    1.  `▶️` Ejecuta el script `extract_reddit_data.py`.
    2.  `📡` Se conecta a la API de Reddit y busca nuevos comentarios sobre Intel y AMD.
    3.  `💾` Actualiza el archivo `data/reddit_comments.csv` con los nuevos hallazgos.
    4.  `⬆️` Hace `commit` y `push` del archivo actualizado de vuelta al repositorio.

#### **🤖 Bot 2: El Procesador y Entrenador (Reactivo)**

* **Activación:** Se dispara **inmediatamente después** de que el Bot 1 sube los nuevos datos.
* **Misión:**
    1.  `▶️` Ejecuta el script `nlp_processor.py`.
    2.  `🧠` Toma los comentarios crudos y los enriquece con análisis de sentimiento, emociones y tópicos, guardando el resultado en `data/processed_reddit_data.csv`.
    3.  `▶️` Ejecuta el script `python_train.py`.
    4.  `🏋️‍♂️` Entrena un nuevo modelo de sentimiento desde cero utilizando **AutoML (PyCaret)**.
    5.  `🎨` Guarda los artefactos del entrenamiento: el modelo (`models/sentiment_model_v2.pkl`) y el gráfico de importancia de características.
    6.  `⬆️` Hace `commit` y `push` de todos los artefactos actualizados al repositorio.

Este ciclo asegura que el proyecto se mantenga siempre al día con la conversación más reciente, mejorando y adaptándose sin intervención manual.

## 💡 La Historia: AutoML vs. Modelo Experto

Uno de los hallazgos clave de este proyecto es la comparación directa entre un modelo entrenado por nosotros y un modelo pre-entrenado de última generación (State-of-the-Art).

* **Nuestro Modelo (AutoML):** Como se puede ver en el dashboard, el modelo entrenado con PyCaret, a pesar de los esfuerzos, obtiene un rendimiento bajo. Esto demuestra empíricamente que el lenguaje de Reddit es demasiado complejo y lleno de matices para los modelos de ML tradicionales basados en frecuencia de palabras.
* **Modelo Experto (RoBERTa):** Para la aplicación final, se toma la decisión estratégica de usar un modelo Transformer pre-entrenado (`cardiffnlp/twitter-roberta-base-sentiment-latest`), que ofrece una precisión muy superior al entender el contexto del lenguaje.

Esta comparación no es un fallo, sino la **demostración de un proceso de MLOps maduro**: saber experimentar, analizar resultados y elegir la mejor herramienta para el producto final.

## 💻 Cómo Ejecutar el Proyecto Localmente

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
    cd TU_REPOSITORIO
    ```

2.  **Crear y Activar un Entorno Virtual:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Instalar las Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar las Credenciales de la API de Reddit:**
    * Crea un archivo llamado `.env` en la raíz del proyecto.
    * Añade tus credenciales dentro de este archivo:
        ```env
        REDDIT_CLIENT_ID="TU_CLIENT_ID"
        REDDIT_CLIENT_SECRET="TU_CLIENT_SECRET"
        REDDIT_USER_AGENT="Script de análisis by TU_USUARIO_REDDIT"
        ```

5.  **Ejecutar el Pipeline en Orden:**
    * **Paso 1: Extraer los datos crudos.**
        ```bash
        python extract_reddit_data.py
        ```
    * **Paso 2: Procesar los datos con los modelos de NLP.** (Este paso puede tardar varios minutos).
        ```bash
        python nlp_processor.py
        ```
    * **Paso 3: Entrenar tu modelo con AutoML.**
        ```bash
        python python_train.py
        ```
    * **Paso 4: Lanzar el Dashboard.**
        ```bash
        python app.py
        ```

6.  Abre la URL local (ej. `http://127.0.0.1:7860`) en tu navegador para ver el dashboard.

## 📁 Estructura del Proyecto


<img width="590" height="434" alt="image" src="https://github.com/user-attachments/assets/d351cd08-88ba-4c4d-8c70-4c53c27d03f6" />
