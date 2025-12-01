import os
import pandas as pd
import numpy as np
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool
from dotenv import load_dotenv

# Cargamos las claves
load_dotenv()

# Esto es una custom tool que lee los csv y te hace un reporte de valores nulos, duplicados, tipos de datos y muestra una muestra
@tool
def inspeccionar_calidad_csv(filepath: str) -> str:
    df = pd.read_csv(filepath)
    nombre_archivo = os.path.basename(filepath)
    target_col = df.columns[-1]

    lines = []

    # Datos generales
    lines.append(f"Reporte general: {nombre_archivo}")
    lines.append(f"\n Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
    lines.append(f"\n Duplicados: {df.duplicated().sum()} filas")
    
    # Columnas categóricas o booleanas
    cols_cat = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist() # guardamos el nombre de las columnas categóricas

    if len(cols_cat) > 0:
        lines.append(f"Se detectaron {len(cols_cat)} columnas categóricas.")
    else:
        lines.append("Todas las columnas son numéricas.")

    # Balanceo de datos
    value_counts = df[target_col].value_counts()
    value_percentages = df[target_col].value_counts(normalize=True) * 100
    lines.append(f"La distribución de datos en la variable objetivo {target_col} es:\n"
                 f"Frecuencia:{value_counts}"
                 f"Porcentaje:{value_percentages}"
                 )
    
    # Datos por columna
    lines.append(f"\n Reporte por columna")
    for col in df.columns:
        # Datos generales
        nulos = df[col].isnull().sum()
        porcentaje_nulos = (nulos / len(df)) * 100
        tipo = df[col].dtype
        uniques = df[col].nunique()
    

        # Outliers
        info_outliers = ""
        if np.issubdtype(tipo, np.number):
            # Cálculo de IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # Contamos cuántos se salen del rango
            cnt_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            
            if cnt_outliers > 0:
                porcentaje_outliers = (cnt_outliers / len(df)) * 100
                info_outliers = f"{cnt_outliers} ({porcentaje_outliers:.1f}%)"
            else:
                info_outliers = "0"

        lines.append(f"Nombre columna={col} | Tipo de dato={tipo} | Nulos={nulos} ({porcentaje_nulos:.1f}%) | Únicos={uniques} | Outliers={info_outliers}")

    lines.append(f"\n 5 primeras filas")
    lines.append(df.head(5).to_string()) # uso to_string para devolver un texto y luego juntar todo el texto de lines=[] con .join

    return "\n".join(lines)

quality_agent = Agent(
    name="Agente de Calidad",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[inspeccionar_calidad_csv], 
    markdown=True,
    instructions=[
        "Eres un experto en Data Quality.",
        "Recibes una solicitud para analizar un archivo.",
        "Usas la herramienta 'inspeccionar_calidad_csv' para ver los datos y generar un reporte.",
        "Indica si hay valores nulos y outliers.",
        "Indica cuántas y cuáles son las columnas categóricas ('cols_cat').",
        "Indica si hay desbalanceo de datos. Consideras un dataset desbalanceado si una clase es < 40%."
    ]
)
