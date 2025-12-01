
# Agno Sistema Multiagente AutoML

Este repositorio contiene un **Pipeline de Machine Learning Automatizado (AutoML)** basado en una arquitectura de **Sistemas Multi-Agente**.

El proyecto utiliza el framework **Agno** y el modelo **Google Gemini 2.5 Flash** para orquestar un equipo de agentes de IA que analizan, limpian, procesan y modelan datos sin intervención humana.

> **Nota:** Este es un **proyecto personal** desarrollado con fines educativos para explorar el potencial de los Agentes de IA en flujos de trabajo de Data Science. No pretende ser una herramienta de producción, sino una prueba de concepto sobre orquestación lógica.

## Estructura

├── agents/               # Lógica de los Agentes (Quality, Strategy, Modeling...)
├── data/
│   ├── raw/              # Ubicación del CSV de entrada
│   ├── processed_data/   # Archivos temporales generados por cada agente
│   └── clean_data/       # Copia final limpia antes del modelado
├── main.py               # Script orquestador principal
├── requirements.txt      # Librerías necesarias
└── .env                  # Variables de entorno

## Funcionalidades

El sistema despliega una cadena de agentes especializados que se comunican entre sí:

1.  **Quality Agent:** Analiza y realiza un reporte de calidad del dataset (Nulos, Outliers, Tipos de datos, Balanceo de datos).
2.  **Strategy Agent (El Director):** Analiza el reporte de calidad y decide la estrategia.
3.  **Imputation Agent:** Gestiona valores nulos según la estrategia definida.
4.  **Outlier Agent:** Detecta y gestiona valores atípicos.
5.  **Encoding Agent:** Transforma variables categóricas a numéricas.
6.  **Modeling Agent:** Normaliza los datos. Entrena un modelo **Random Forest**, aplica balanceo SMOTE si es necesario y evalúa métricas (Accuracy, F1, ROC-AUC, Matriz de Confusión).


## Gestión de archivos csv

El Sistema Multiagente ha sido probado principalmente con el archivo "Bullying1.csv" que aparece en "data/raw/Bullying1.csv". Si se quiere probar con otro dataset, simplemente hay que subir un nuevo .csv en la carpeta "data/raw" y cambiar la variable archivo_objetivo en main.py.

Durante el proceso de AutoML, el Sistema Multiagente creará diferentes .csv con los diferentes preprocesamientos aplicados y todos almacenados en una carpeta "data/processed_data".

Al final del preprocesamiento, guardaremos la última copia del csv. procesado previa a la aplicación de balanceo de datos y modelado en "data/clean_data".

Todo con el objetivo de comprobar el avance de nuestro Sistema Multiagente. Para evitar posibles errores al volver a ejecutar "python main.py" el sistema buscará estas dos carpetas, 
las borrará junto a todo su contenido y volverá a crearlas vacías.

## Requisitos Previos

Toda la instalación previa requerida para ejecutar este sistema multiagente se encuentra en requirements.txt.

## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/MiguelPastorB/SM-project.git
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configurar entorno:**
    Crea un archivo `.env` en la raíz del proyecto y añade tu clave:
    ```env
    GOOGLE_API_KEY=tu_clave_secreta_aqui
    ```

## Ejecución y Uso

Para iniciar el sistema multiagente, ejecuta:

```bash
python main.py

## Problemática con lincencia gratuita de Gemini

Dado que este proyecto utiliza la versión gratuita de la API de Google Gemini, el script incluye pausas intencionales (sleep) entre agentes para evitar errores de saturación (429 Too Many Requests).

Debido a esta restricción, la complejidad de los agentes y la variedad de modelos de ML implementados se ha mantenido acotada para asegurar la estabilidad del flujo.