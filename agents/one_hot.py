import pandas as pd
import os
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool
from dotenv import load_dotenv

# Cargamos las claves
load_dotenv()

# Creamos la herramienta que aplica One-hot encoding (dummies)
# bool = True quita una de las 2 columnas booleanas creadas por One-hot encoding
# Guarda el resultado en 'data/processed_data'
@tool
def aplicar_dummies(filepath: str) -> str:

    df = pd.read_csv(filepath)
    
    # Identificar columnas categóricas
    cols_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

    df_final = df.drop(columns=cols_categoricas) # cojo solo las columnas numéricas

    # Aplicamos one-hot encoding a cada columnas categórica y le devolvemos el nombre original (no funciona con multiclase)
    for col in cols_categoricas:
        dummies = pd.get_dummies(df[col], drop_first=True, dtype=int)
        dummies.columns = [col]
        
        # Unimos la columnas a las columnas numéricas de nuestro dataset
        df_final = pd.concat([df_final, dummies], axis=1)

    # Ruta de destino
    output_folder = "data/processed_data"

    base_name = os.path.basename(filepath)
    file_name_sin_ext = os.path.splitext(base_name)[0]

    # Quitamos sufijos anteriores para no hacer nombres de csv largos
    clean_name = file_name_sin_ext.replace("_no_nulls", "").replace("_no_outliers", "") # outliers o nulos suelen ser pasos previos
        
    nombre_salida = f"{clean_name}_encoded.csv"
    path_salida = os.path.join(output_folder, nombre_salida)

    # Guardamos
    df_final.to_csv(path_salida, index=False)
    
    # Calcular cuántas columnas nuevas nacieron
    cols_antes = df.shape[1]
    cols_despues = df_final.shape[1]
    nuevas_cols = cols_despues - cols_antes

    return (
        f"### Transformación a datos numérico (dummies) completada\n"
        f"- **Columnas originales:** {cols_antes}\n"
        f"- **Columnas finales:** {cols_despues} (Crecimiento: +{nuevas_cols})\n"
        f"- **Variables transformadas:** {cols_categoricas}\n"
        f"- **Archivo guardado en:** `{path_salida}`"
    )


# Agente
one_hot_agent = Agent(
    name="Agente de One-hot Encoding",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[aplicar_dummies],
    markdown=True,
    instructions=[
        "Eres un ingeniero de datos experto en preprocesamiento.",
        "Tu objetivo es preparar los datos para que sean 100% numéricos.",
        "Recibe un archivo, aplica 'aplicar_dummies' y reporta el cambio de dimensiones y los resultados."
    ]
)