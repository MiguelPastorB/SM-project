import pandas as pd
import numpy as np
import os
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool
from dotenv import load_dotenv

# Cargamos las claves
load_dotenv()

# Esta herramienta detecta y gestiona valores atípicos (outliers) usando el método del Rango Intercuartílico (IQR)
# Guarda el resultado en 'data/processed_data'
@tool
def gestionar_outliers(filepath: str, estrategia: str = "eliminar", columna: str = "all") -> str:

    df = pd.read_csv(filepath)
    filas_iniciales = len(df)
    
    # El agente que dirige dirá qué columnas utilizamos
    if columna == "all": # si utilizamos todas las columnas las guardamos
        # Solo columnas numéricas, el IQR no funciona con texto
        cols_to_check = df.select_dtypes(include=[np.number]).columns.tolist()
    elif columna in df.columns: # si utilizamos columnas específicas las guardamos
        cols_to_check = [columna]
    else:
        return f"Error: La columna '{columna}' no existe en el archivo."

    total_outliers_detectados = 0
        
    # Bucle de detección y corrección (Método IQR)
    for col_name in cols_to_check:
        # Calcular cuartiles
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir límites
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Máscara de outliers en esta columna
        mask_outliers = (df[col_name] < lower_bound) | (df[col_name] > upper_bound) # true=outlier / false=no outlier
        num_outliers = mask_outliers.sum() # suma los true
        
        if num_outliers > 0:
            total_outliers_detectados += num_outliers
            
            if estrategia.lower() == "eliminar":
                # Nos quedamos con lo que no es outlier
                df = df[~mask_outliers] # ~ significa NO, es decir quita los outliers
            
            elif estrategia.lower() == "capping":
                # Si es > max, se vuelve max. Si es < min, se vuelve min.
                df.loc[df[col_name] < lower_bound, col_name] = lower_bound
                df.loc[df[col_name] > upper_bound, col_name] = upper_bound

    # Guardado de destino
    output_folder = "data/processed_data"

    clean_name = os.path.basename(filepath).split("_")[0] if "_" in os.path.basename(filepath) else os.path.basename(filepath).split(".")[0]
    
    nombre_salida = f"{clean_name}_no_outliers.csv"
    path_salida = os.path.join(output_folder, nombre_salida)
    
    # Guardar
    df.to_csv(path_salida, index=False)
    
    # Datos para el reporte
    filas_finales = len(df)
    filas_perdidas = filas_iniciales - filas_finales
    
    resumen_accion = ""
    if estrategia == "eliminar":
        resumen_accion = f"Se eliminaron **{filas_perdidas}** filas."
    else:
        resumen_accion = f"Se suavizaron los valores extremos. Filas mantenidas: {filas_finales}."

    return (
        f"### Reporte de Outliers (Método IQR)\n"
        f"- **Columnas analizadas:** {len(cols_to_check)}\n"
        f"- **Estrategia:** {estrategia.upper()}\n"
        f"- **Outliers detectados (Total):** {total_outliers_detectados}\n"
        f"- **Acción:** {resumen_accion}\n"
        f"- **Archivo guardado en:** `{path_salida}`"
    )


# Agente
outlier_agent = Agent(
    name="Agente de Outliers",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[gestionar_outliers],
    markdown=True,
    instructions=[
        "Eres un experto estadístico encargado de limpiar datos atípicos.",
        "Tu herramienta principal es 'gestionar_outliers'.",
        "Si el usuario no especifica qué hacer, usa la estrategia 'eliminar' por defecto.",
        "Si el usuario pide 'suavizar' o 'mantener datos', usa la estrategia 'capping'."
        "Reporta el cambio de dimensiones y resultados."
    ]
)