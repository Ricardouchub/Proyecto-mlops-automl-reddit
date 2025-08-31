#  Pipeline de MLOps para Análisis de Sentimiento en Tiempo Real

### Análisis Competitivo Automatizado: Intel vs. AMD en Reddit

**[➡️ Ver Demo en Vivo](https://TU_URL_DE_RENDER.onrender.com)**

---

Este proyecto demuestra la construcción de un **pipeline de MLOps de extremo a extremo**, diseñado para ser completamente autónomo. El sistema extrae datos en vivo desde la API de Reddit, los procesa utilizando múltiples modelos de NLP, entrena un modelo de Machine Learning propio con AutoML, y despliega los resultados en un dashboard interactivo que se actualiza automáticamente.

## Características Principales

* **Extracción de Datos Automatizada:** Un bot se conecta diariamente a Reddit para recolectar nuevos comentarios sobre Intel y AMD.
* **Procesamiento Avanzado de NLP:** Cada comentario es enriquecido con un análisis multifacético:
    * **Sentimiento:** Positivo, Negativo o Neutral.
    * **Emociones:** Alegría, Ira, Miedo, etc.
    * **Modelado de Tópicos:** Descubrimiento de los principales temas de conversación.
* **Entrenamiento con AutoML y GPU:** Se entrena un modelo de clasificación desde cero utilizando **PyCaret**, con la opción de aceleración por GPU.
* **Dashboard Comparativo:** Una interfaz interactiva construida con **Gradio** que permite comparar el rendimiento del modelo propio (AutoML) contra un modelo experto de última generación (RoBERta).
* **Orquestación con GitHub Actions:** Múltiples pipelines de CI/CD que automatizan todo el flujo, desde la extracción de datos hasta el re-entrenamiento del modelo.
* **Despliegue Continuo (CI/CD) en Render:** La aplicación está conectada al repositorio de GitHub, desplegándose automáticamente con cada nueva actualización del modelo.

##  El Pipeline de MLOps en Acción

Este proyecto es un **sistema vivo y autónomo**. La magia reside en la orquestación de workflows de GitHub Actions que trabajan en conjunto:

#### ** Bot 1: El Extractor de Datos (Diario)**

* **Activación:** Se ejecuta automáticamente todos los días a las 05:00 UTC.
* **Misión:**
    1.  Ejecuta el script `extract_reddit_data.py`.
    2.  Se conecta a la API de Reddit y busca nuevos comentarios sobre Intel y AMD.
    3.  Actualiza el archivo `data/reddit_comments.csv`.
    4.  Hace `commit` y `push` del archivo actualizado al repositorio, **activando el siguiente bot**.

#### ** Bot 2: El Procesador y Entrenador (Reactivo)**

* **Activación:** Se dispara inmediatamente después de que el Bot 1 sube los nuevos datos.
* **Arquitectura:** Para solucionar el problema de `No space left on device` en los runners de GitHub, este pipeline se divide en **dos trabajos secuenciales y especializados**:
    1.  **Job `process-data`:**
        *  Instala **solo** las librerías de NLP y ejecuta `nlp_processor.py` para enriquecer los datos.
        *  Guarda los datos procesados como un "artefacto" temporal.
    2.  **Job `train-model`:**
        * Descarga el artefacto del job anterior.
        *  Instala **solo** las librerías de AutoML y ejecuta `python_train.py` para entrenar el modelo.
        *  Guarda el modelo (`.pkl`) y el gráfico de importancia como artefactos finales.

##  La Historia: AutoML vs. Modelo Experto

Uno de los hallazgos clave de este proyecto es la comparación directa entre un modelo entrenado por nosotros y un modelo pre-entrenado de última generación.

* **Nuestro Modelo (AutoML):** Como se ve en el dashboard, el modelo entrenado con PyCaret logra un rendimiento moderado, demostrando que ha aprendido patrones reales pero que el lenguaje de Reddit es un desafío complejo.

    **Resultados del Mejor Modelo (ExtraTreesClassifier):**
    * **Accuracy:** `0.5884` (significativamente mejor que el azar del 33%).
    * **AUC:** `0.7435` (buena capacidad para discriminar entre clases).
    * **Kappa:** `0.2786` (rendimiento bajo-moderado por encima del azar).

* **Modelo Experto (RoBERTa):** Para el análisis principal en el dashboard, se utiliza un modelo Transformer pre-entrenado (`cardiffnlp/twitter-roberta-base-sentiment-latest`), que ofrece una precisión muy superior al entender el contexto profundo del lenguaje.

Esta comparación no es un fallo, sino la **demostración de un proceso de MLOps maduro**: saber experimentar, analizar resultados y elegir la mejor herramienta para el producto final.

# Cómo Ejecutar el Proyecto Localmente

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
    * **Paso 2: Procesar los datos con los modelos de NLP.** (Este paso puede tardar).
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

6.  Abre la URL local (ej. `http://127.0.0.1:7860`) en tu navegador.
