import os
import pandas as pd
import numpy as np
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# This tool evaluates the quality of a CSV file and generates a detailed report
@tool
def evaluate_csv_quality(filepath: str) -> str:
    # Read the CSV file and handle potential errors
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return f"Error: El archivo '{filepath}' no fue encontrado."
    except pd.errors.EmptyDataError:
        return f"Error: El archivo '{filepath}' está vacío."
    except Exception as e:
        return f"Error leyendo el archivo: {e}"
    # Extract file name for reporting
    file_name = os.path.basename(filepath)
    # We assume the target variable is the last column
    target_col = df.columns[-1]

    # Start building the report
    report = ""
    report += f"Reporte general: {file_name}"
    report += f"\n Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas"
    report += f"\n Duplicados: {df.duplicated().sum()} filas"
    # Identify categorical columns
    cols_cat = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # Report on categorical columns
    if len(cols_cat) > 0:
        report += f"Se detectaron {len(cols_cat)} columnas categóricas."
    else:
        report += "Todas las columnas son numéricas."

    # Check for class imbalance in the target variable
    value_counts = df[target_col].value_counts()
    value_percentages = df[target_col].value_counts(normalize=True) * 100
    report += f"La distribución de datos en la variable objetivo {target_col} es:\n"
    report += f"Frecuencia:{value_counts}"
    report += f"Porcentaje:{value_percentages}"

    # Report on nulls and outliers per column
    report += f"\n Reporte por columna"
    for col in df.columns:
        nulls = df[col].isnull().sum() # Count nulls
        percentage_nulls = (nulls / len(df)) * 100
        data_type = df[col].dtype # Data type
        uniques = df[col].nunique() # Unique values count
    
        # Outlier detection using IQR method for numeric columns
        info_outliers = ""
        if np.issubdtype(data_type, np.number):
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            # Count outliers and percentage for the column
            cnt_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if cnt_outliers > 0:
                percentage_outliers = (cnt_outliers / len(df)) * 100
                info_outliers = f"{cnt_outliers} ({percentage_outliers:.1f}%)"
            else:
                info_outliers = "0"
        # Build column report line
        report += f"Nombre columna={col} | Tipo de dato={data_type} | Nulos={nulls} ({percentage_nulls:.1f}%) | Únicos={uniques} | Outliers={info_outliers}"

    return report

# Agent
quality_agent = Agent(
    name="Agente de Calidad",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[evaluate_csv_quality], 
    markdown=True,
    instructions=[
        "Eres un experto en Data Quality.",
        "Recibes una solicitud para analizar un archivo.",
        "Usas la herramienta 'evaluate_csv_quality' para ver los datos y generar un reporte.",
        "Indica si hay valores nulos y outliers.",
        "Indica cuántas y cuáles son las columnas categóricas ('cols_cat').",
        "Indica si hay desbalanceo de datos. Consideras un dataset desbalanceado si una clase es < 40%."
    ]
)
