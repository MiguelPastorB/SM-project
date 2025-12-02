import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool
import os
from dotenv import load_dotenv

# Cargamos las claves
load_dotenv()

# Creamos una herramienta personalizada donde aplicamos técnicas de limpieza de valores nulos (NaN) sobre un archivo CSV
# y guardamos el resultado en 'data/processed_data'. El agente principal elegirá si eliminar o rellenar datos y también
# elegirá el valor de k para aplicar KNN.
@tool
def aplicar_imputacion(filepath: str, estrategia: str = "eliminar") -> str:

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return f"Error: El archivo '{filepath}' no fue encontrado."
    except pd.errors.EmptyDataError:
        return f"Error: El archivo '{filepath}' está vacío."
    except Exception as e:
        return f"Error leyendo el archivo: {e}"
    
    # Definimos la carpeta de destino
    output_folder = os.path.join("data", "processed_data")
        
    # Construimos el nuevo nombre
    clean_name = os.path.basename(filepath).split("_")[0] if "_" in os.path.basename(filepath) else os.path.basename(filepath).split(".")[0]
    
    # Nombre final: data/processed_data/Bullying1_no_nulls.csv
    nombre_salida = f"{clean_name}_no_nulls.csv"
    path_salida = os.path.join(output_folder, nombre_salida)
    
    filas_iniciales = len(df)
        
    try:
        # Estrategia de limpieza
        if estrategia.lower() == "eliminar":
            df_clean = df.dropna()
            filas_borradas = filas_iniciales - len(df_clean)
            
            # Guardamos en la nueva ruta
            df_clean.to_csv(path_salida, index=False)
            
            if filas_borradas == 0:
                return f"Sin nulos. Copia guardada en: {path_salida}"
            return f"Se eliminaron {filas_borradas} filas. Dataset limpio en: {path_salida}"

        # Estrategia de rellenar datos con KNN   
        elif estrategia.lower() == "knn":
            df_numeric = df.select_dtypes(include=[np.number]) # columnas numéricas para KNN
            df_categorical = df.select_dtypes(exclude=[np.number]) # columnas categóricas
            
            if df_numeric.empty:
                return "No hay columnas numéricas para aplicar KNN."

            imputer = KNNImputer(n_neighbors=5) # instanciamos el imputador
            matriz_imputada = imputer.fit_transform(df_numeric) # ejecutamos todo el proceso y devuelve una matriz
            
            df_numeric_imputed = pd.DataFrame( # transformamos la matriz en un dataframe de nuevo y volvemos a poner los índices y columnas previos
                matriz_imputada, columns=df_numeric.columns, index=df_numeric.index
            )
            
            df_final = pd.concat([df_numeric_imputed, df_categorical], axis=1) # volvemos a juntar todas las columnas
            
            # Comprobamos que no quedan nulos
            total_nulos_restantes = df_final.isnull().sum()

            # Guardamos en la nueva ruta
            try:
                df_final.to_csv(path_salida, index=False)
            except Exception as e:
                return f"Error guardando el archivo procesado: {e}"
            
            if total_nulos_restantes.sum() == 0:
                return f"Imputación KNN realizada exitosamente. No quedan nulos.\nArchivo guardado en: {path_salida}"
            else:
                return (f"Imputación KNN realizada pero quedan {total_nulos_restantes} valores nulos.\n"
                        f"Archivo guardado en: {path_salida}")
        else:
            return ("Estrategia no reconocida. Usa 'eliminar' o 'knn'.")
    except Exception as e:
        return f"Error durante la imputación: {e}"
        

# Agente
nan_imputer_agent = Agent(
    name="Agente de Imputación",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[aplicar_imputacion],
    markdown=True,
    instructions=[
        "Recibes un archivo.",
        "Aplicas la estrategia de limpieza indicada.",
        "Informas que el resultado se ha guardado en 'data/processed_data'."
        "Reporta el cambio de dimensiones y resultados."
    ]
)