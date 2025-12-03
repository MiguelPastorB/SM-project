import pandas as pd
import numpy as np
import os
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# This tool manages outliers in a CSV file using specified strategies
@tool
def manage_outliers(filepath: str, strategy: str = "drop", column: str = "all") -> str:
    # Read the CSV file and handle potential errors
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return f"Error: El archivo '{filepath}' no fue encontrado."
    except pd.errors.EmptyDataError:
        return f"Error: El archivo '{filepath}' está vacío."
    except Exception as e:
        return f"Error leyendo el archivo: {e}"
    
    # Store initial number of rows for reporting
    initial_rows = len(df)
    
    # Validate strategy
    if strategy.lower() not in ["drop", "capping"]:
        return f"Estrategia '{strategy}' no soportada. Usa 'drop' o 'capping'."
    
    # Determine columns to process
    if column == "all":
        cols_to_check = df.select_dtypes(include=[np.number]).columns.tolist() # All numeric columns
        if not cols_to_check:
            return "No hay columnas numéricas para analizar outliers."
    elif column in df.columns:
        # Check if the specified column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"La columna '{column}' no es numérica y no puede analizarse con IQR."
        cols_to_check = [column]
    else:
        return f"Error: La columna '{column}' no existe en el archivo."

    # Initialize outlier counter
    total_outliers_found = 0
        
    try:
        # Process each column and manage outliers
        for col_name in cols_to_check:
            # Calculate IQR
            Q1 = df[col_name].quantile(0.25)
            Q3 = df[col_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            mask_outliers = (df[col_name] < lower_bound) | (df[col_name] > upper_bound)
            num_outliers = mask_outliers.sum()
            
            # if outliers found, take action. If none, skip
            if num_outliers > 0:
                total_outliers_found += num_outliers
                
                if strategy.lower() == "drop":
                    
                    df = df[~mask_outliers] # Drop outlier rows (~ is negation)
                
                elif strategy.lower() == "capping":
                    
                    df.loc[df[col_name] < lower_bound, col_name] = lower_bound
                    df.loc[df[col_name] > upper_bound, col_name] = upper_bound
    except Exception as e:
        return f"Error durante la gestión de outliers: {e}"

    # Save the processed DataFrame
    output_folder = os.path.join("data", "processed_data")
    clean_name = os.path.basename(filepath).split("_")[0] if "_" in os.path.basename(filepath) else os.path.basename(filepath).split(".")[0]
    output_name = f"{clean_name}_no_outliers.csv"
    output_path = os.path.join(output_folder, output_name)
    
    # Attempt to save the DataFrame to a CSV file
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        return f"Error guardando el archivo procesado: {e}"
    
    
    final_rows = len(df)
    lost_rows = initial_rows - final_rows
    
    action_summary = ""
    if strategy == "drop":
        action_summary = f"Se eliminaron **{lost_rows}** filas."
    else:
        action_summary = f"Se suavizaron los valores extremos. Filas mantenidas: {final_rows}."

    return (
        f"### Reporte de Outliers (Método IQR)\n"
        f"- **Columnas analizadas:** {len(cols_to_check)}\n"
        f"- **Estrategia:** {strategy.upper()}\n"
        f"- **Outliers detectados (Total):** {total_outliers_found}\n"
        f"- **Acción:** {action_summary}\n"
        f"- **Archivo guardado en:** `{output_path}`"
    )


# Agent
outlier_agent = Agent(
    name="Agente de Outliers",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[manage_outliers],
    markdown=True,
    instructions=[
        "Eres un experto estadístico encargado de limpiar datos atípicos.",
        "Tu herramienta principal es 'manage_outliers'.",
        "Si el usuario no especifica qué hacer, usa la estrategia 'drop' por defecto.",
        "Si el usuario pide 'suavizar' o 'mantener datos', usa la estrategia 'capping'."
        "Reporta el cambio de dimensiones y resultados."
    ]
)