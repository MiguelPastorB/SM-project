import pandas as pd
import os
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@tool
def apply_dummies(filepath: str) -> str:
    # Read the CSV file and handle potential errors
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return f"Error: El archivo '{filepath}' no fue encontrado."
    except pd.errors.EmptyDataError:
        return f"Error: El archivo '{filepath}' está vacío."
    except Exception as e:
        return f"Error leyendo el archivo: {e}"
    
    # Identify categorical columns
    cols_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cols_cat:
        return "No se encontraron columnas categóricas para transformar."
    # Initialize final DataFrame without categorical columns
    df_final = df.drop(columns=cols_cat)

    # Apply one-hot encoding
    try:
        # Create dummies for each categorical column and concatenate to final DataFrame
        for col in cols_cat:
            dummies = pd.get_dummies(df[col], drop_first=True, dtype=int)
            dummies.columns = [col]
            df_final = pd.concat([df_final, dummies], axis=1)
    except Exception as e:
        return f"Error durante la transformación a variables numéricas: {e}"

    # Define output folder and path
    output_folder = os.path.join("data", "processed_data")
    clean_name = os.path.basename(filepath).split("_")[0] if "_" in os.path.basename(filepath) else os.path.basename(filepath).split(".")[0]
    output_name = f"{clean_name}_encoded.csv"
    output_path = os.path.join(output_folder, output_name)

    # Save the final DataFrame
    try:
        df_final.to_csv(output_path, index=False)
    except Exception as e:
        return f"Error guardando el archivo procesado: {e}"
    
    # Report on dimension changes
    cols_before = df.shape[1]
    cols_after = df_final.shape[1]
    new_cols = cols_after - cols_before

    return (
        f"### Transformación a datos numérico (dummies) completada\n"
        f"- **Columnas originales:** {cols_before}\n"
        f"- **Columnas finales:** {cols_after} (Crecimiento: +{new_cols})\n"
        f"- **Variables transformadas:** {cols_cat}\n"
        f"- **Archivo guardado en:** `{output_path}`"
    )


# Agent
one_hot_agent = Agent(
    name="Agente de One-hot Encoding",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[apply_dummies],
    markdown=True,
    instructions=[
        "Eres un ingeniero de datos experto en preprocesamiento.",
        "Tu objetivo es preparar los datos para que sean 100% numéricos.",
        "Recibe un archivo, aplica 'apply_dummies' y reporta el cambio de dimensiones y los resultados."
    ]
)