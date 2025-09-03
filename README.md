#   MLOps y AutoML: Pipeline de Análisis de Sentimiento para comparar Intel vs. AMD

<p align="left">
  <img src="https://img.shields.io/badge/Estado-Operativo-2ECC71?style=flat-square&logo=checkmarx&logoColor=white" alt="Estado: Operativo"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Reddit_API-Fuente_de_Datos-FF4500?style=flat-square&logo=reddit&logoColor=white" alt="Reddit API"/>
  <img src="https://img.shields.io/github/actions/workflow/status/Ricardouchub/proyecto-mlops-reddit/.github/workflows/extract_data_schedule.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI%2FCD" alt="CI/CD"/>
  <img src="https://img.shields.io/badge/Plotly-Visualizacion-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly"/>
  <img src="https://img.shields.io/badge/AutoML-Clasificador-00C853?style=flat-square" alt="AutoML"/>
  <img src="https://img.shields.io/badge/Render-Despliegue-46E3B7?style=flat-square&logo=render&logoColor=white" alt="Render"/>
</p>


Construcción de un **pipeline de MLOps de extremo a extremo**, diseñado para ser completamente autónomo. 

El sistema extrae datos en vivo desde la API de Reddit, los procesa, entrena un modelo de Machine Learning propio con **AutoML** y despliega los resultados en un dashboard interactivo que se actualiza automáticamente.

<div align="center">

[Resultados del Modelo AutoML](https://proyecto-mlops-reddit.onrender.com)

<img width="570" height="499" alt="image" src="https://github.com/user-attachments/assets/007d5df5-7aa5-4810-9040-577a0c8f5a58" />

</div>

---


## Características

* **Extracción de Datos Automatizada:** Un bot se conecta diariamente a Reddit para recolectar nuevos comentarios sobre Intel y AMD.
* **Procesamiento Avanzado de NLP:** Cada comentario es etiquetado automáticamente con Sentimiento y Emociones usando un modelo Transformer pre-entrenado.
* **Entrenamiento con AutoML y GPU:** Se entrena un modelo de clasificación desde cero utilizando **PyCaret**, demostrando un pipeline de entrenamiento automatizado que puede ser acelerado por GPU.
* **Dashboard Interactivo:** Una interfaz construida con **Gradio** que visualiza los resultados del modelo entrenado, permitiendo filtrar y explorar los datos.
* **Orquestación con GitHub Actions:** Múltiples pipelines de CI/CD que automatizan todo el flujo, desde la extracción de datos hasta el re-entrenamiento del modelo y la generación de resultados.
* **Despliegue Continuo (CI/CD) en Render:** La aplicación está conectada al repositorio de GitHub, desplegándose automáticamente con cada nueva actualización.

---
##  Pipeline de MLOps

Este proyecto es un sistema vivo y autónomo. La magia reside en la **orquestación de workflows de GitHub Actions** que trabajan en conjunto:

#### **Bot 1: Extractor de Datos Diario**

* **Activación:** Se ejecuta automáticamente todos los días a las 05:00 UTC.
* **Misión:**
    1. Ejecuta el script `extract_reddit_data.py`.
    2. Se conecta a la API de Reddit y busca nuevos comentarios sobre Intel y AMD.
    3. Actualiza el archivo de datos crudos `data/reddit_comments.csv`.
    4. Hace `commit` y `push` del archivo actualizado al repositorio, **activando el siguiente bot**.

#### **Bot 2: Procesador y Entrenador**

* **Activación:** Se dispara automáticamente cuando el Bot 1 termina con éxito.
* **Arquitectura:** Para solucionar problemas de memoria (`No space left on device`), este pipeline se divide en **dos trabajos secuenciales**:
    1.  **Job `process-data`:**
        * Instala las librerías de NLP y ejecuta `nlp_processor.py` para enriquecer los datos (sentimiento, emociones, etc.).
        * Guarda los datos procesados como un "artefacto" temporal.
    2.  **Job `train-model`:**
        * Descarga el artefacto del job anterior.
        * Instala las librerías de AutoML y ejecuta `python_train.py`. Este script entrena el modelo y, crucialmente, **genera el archivo de resultados `data/automl_results.csv`**.
        * Hace `commit` y `push` de todos los artefactos finales (el modelo `.pkl`, la info del modelo `.txt` y los resultados `.csv`), **activando el despliegue en Render.**

---
## Conclusiones del Modelo AutoML

El objetivo de este proyecto es demostrar un pipeline funcional, y los resultados del entrenamiento son un hallazgo clave. A pesar de la complejidad y el "ruido" del lenguaje de Reddit, el modelo AutoML logró un rendimiento moderado, demostrando que ha aprendido patrones reales en los datos.

**Resultados del Mejor Modelo (ExtraTreesClassifier):**
* **Accuracy:** `0.5884`
* **AUC:** `0.7435`
* **Kappa:** `0.2786` 

Estos resultados realistas son una excelente demostración de un problema de NLP del mundo real. Muestran que, si bien el modelo tiene poder predictivo, hay un amplio margen de mejora, lo que justifica la existencia del pipeline para re-entrenar y mejorar el modelo a medida que se recolectan más datos.


---

## Estructura del Proyecto

```bash
├── .github/workflows/         # Contiene los pipelines de GitHub Actions
│   ├── extract_data_schedule.yml
│   └── process_and_train.yml
├── data/                      # Almacena los datasets generados
│   ├── reddit_comments.csv
│   ├── processed_reddit_data.csv
│   └── automl_results.csv
├── models/                    # Almacena los artefactos del modelo
│   ├── sentiment_model_v2.pkl
│   └── model_info.txt
├── .gitignore                 
├── app.py                     # El código del dashboard de Gradio
├── extract_reddit_data.py     # Script para extraer datos de Reddit
├── nlp_processor.py           # Script para el análisis NLP de los datos
├── python_train.py            # Script para el entrenamiento y la predicción con AutoML
└── requirements.txt           # Lista de dependencias de Python
```

---
## Autor

**Ricardo Urdnaeta**

