import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# This tool applies missing value imputation strategies to a CSV file
@tool
def manage_nulls(filepath: str, strategy: str = "drop") -> str:
    # Read the CSV file and handle potential errors
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return f"Error: El archivo '{filepath}' no fue encontrado."
    except pd.errors.EmptyDataError:
        return f"Error: El archivo '{filepath}' está vacío."
    except Exception as e:
        return f"Error leyendo el archivo: {e}"
    
    # Define output folder
    output_folder = os.path.join("data", "processed_data")
    # Extract clean name for output file and path
    clean_name = os.path.basename(filepath).split("_")[0] if "_" in os.path.basename(filepath) else os.path.basename(filepath).split(".")[0]
    output_filename = f"{clean_name}_no_nulls.csv"
    destination_path = os.path.join(output_folder, output_filename)
    # Store initial number of rows for reporting
    initial_rows = len(df)
        
    try:
        # Apply the selected strategy
        if strategy.lower() == "drop":
            df_clean = df.dropna() # Drop rows with any null values
            deleted_rows = initial_rows - len(df_clean)
            df_clean.to_csv(destination_path, index=False) # Save cleaned DataFrame
            if deleted_rows == 0: # If no rows were deleted, inform accordingly
                return f"Sin nulos. Copia guardada en: {destination_path}"
            return f"Se eliminaron {deleted_rows} filas. Dataset limpio en: {destination_path}"

        elif strategy.lower() == "knn":
            df_numeric = df.select_dtypes(include=[np.number]) # Select only numeric columns
            df_categorical = df.select_dtypes(exclude=[np.number]) # Select non-numeric columns
            if df_numeric.empty: # Check if there are numeric columns to impute
                return "No hay columnas numéricas para aplicar KNN."
            # Apply KNN Imputer
            imputer = KNNImputer(n_neighbors=5)
            imputed_matrix = imputer.fit_transform(df_numeric)
            df_numeric_imputed = pd.DataFrame(imputed_matrix, columns=df_numeric.columns, index=df_numeric.index)
            df_final = pd.concat([df_numeric_imputed, df_categorical], axis=1) # Combine numeric and categorical data
            # We check if any nulls remain
            total_nulls_remaining = df_final.isnull().sum()
            # Save the final DataFrame
            try:
                df_final.to_csv(destination_path, index=False)
            except Exception as e:
                return f"Error guardando el archivo procesado: {e}"
            # Report on remaining nulls
            if total_nulls_remaining.sum() == 0:
                return f"Imputación KNN realizada exitosamente. No quedan nulos.\nArchivo guardado en: {destination_path}"
            else:
                return (f"Imputación KNN realizada pero quedan {total_nulls_remaining} valores nulos.\n"
                        f"Archivo guardado en: {destination_path}")
        else:
            return ("Estrategia no reconocida. Usa 'drop' o 'knn'.")
    except Exception as e:
        return f"Error durante la imputación: {e}"
        

# Agent
nan_imputer_agent = Agent(
    name="Agente de Imputación",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[manage_nulls],
    markdown=True,
    instructions=[
        "Recibes un archivo.",
        "Tu herramienta principal es 'manage_nulls'.",
        "Aplicas la estrategia de limpieza indicada.",
        "Informas que el resultado se ha guardado en 'data/processed_data'."
        "Reporta el cambio de dimensiones y resultados."
    ]
)