import os
import shutil
from agents.quality import quality_agent
from agents.nan_imputer import nan_imputer_agent
from agents.outliers import outlier_agent
from agents.one_hot import one_hot_agent
from agents.modeling import modeling_agent
from agents.director import strategy_agent
from utils.utils import ejecutar_con_retry
from utils.utils import limpiar_datos_antiguos  
from dotenv import load_dotenv

# Cargamos las claves
load_dotenv()

# Estructura del sistema multiagente
def main():
    # Limpiamos datos antiguos
    limpiar_datos_antiguos()

    archivo_objetivo = "data/raw/Bullying1.csv"
    carpeta_processed = "data/processed_data"
    archivo_actual = archivo_objetivo # la idea de archivo_actual es permitir a los agentes saltarse pasos de preprocesamiento sin crear csv nuevos

    # Paso 1: reporte general
    prompt_quality_report = f"Hazme un reporte de calidad de datos del archivo {archivo_objetivo}"
    quality_report = ejecutar_con_retry("run", quality_agent, prompt_quality_report) # guardamos el reporte

    reporte_texto = quality_report.content # guardamos solo el texto del reporte
    print(reporte_texto) # mostramos el reporte

    # Informamos al agente director
    prompt_director = f"Aquí tienes el reporte técnico:\n{reporte_texto}\nGenera el JSON de decisiones."
    reporte_director = ejecutar_con_retry("run", strategy_agent, prompt_director) # enviamos el reporte al agente director
    plan = dict(reporte_director.content) # obtenemos el diccionario directamente del esquema

    # Paso 2: valores nulos
    accion = plan.get("estrategia_nulos", "saltar")
    prompt_nan = f"Limpia el archivo {archivo_actual} con '{accion}'"
    if accion != "saltar":
        ejecutar_con_retry("print_response", nan_imputer_agent, prompt_nan)
        
        # Actualizamos el csv
        clean_name = os.path.basename(archivo_actual).replace(".csv", "").replace("_no_nulls", "").replace("_no_outliers", "").replace("_encoded", "").replace("_scaled", "")
        nuevo = os.path.join(carpeta_processed, f"{clean_name}_no_nulls.csv")
        if os.path.exists(nuevo): # si creamos un nuevo csv actualizamos archivo_actual sino no lo actualizamos
            archivo_actual = nuevo 
    else:
        print(f"El archivo {archivo_actual} no tiene valores nulos.")

    # Paso 3: outliers
    accion = plan.get("estrategia_outliers", "saltar")
    prompt_outlier = f"Detecta y gestiona outliers en {archivo_actual} con '{accion}'"
    if accion != "saltar":
        ejecutar_con_retry("print_response", outlier_agent, prompt_outlier)
        
        # Actualizamos el csv
        clean_name = os.path.basename(archivo_actual).replace(".csv", "").replace("_no_nulls", "").replace("_no_outliers", "").replace("_encoded", "").replace("_scaled", "")
        nuevo = os.path.join(carpeta_processed, f"{clean_name}_no_outliers.csv")
        if os.path.exists(nuevo):
            archivo_actual = nuevo
    else:
        print(f"El archivo {archivo_actual} no tiene outliers.")

    # Paso 4: One-hot encoding
    accion = plan.get("estrategia_encoding", "saltar")
    prompt_one_hot = f"Aplica transformación numérica (dummies) al archivo {archivo_actual}"
    if accion == "get_dummies":
        ejecutar_con_retry("print_response", one_hot_agent, prompt_one_hot)

        clean_name = os.path.basename(archivo_actual).replace(".csv", "").replace("_no_nulls", "").replace("_no_outliers", "").replace("_encoded", "").replace("_scaled", "")
        nuevo = os.path.join(carpeta_processed, f"{clean_name}_encoded.csv")
        if os.path.exists(nuevo):
            archivo_actual = nuevo
    else:
        print(f"El archivo {archivo_actual} no tiene columnas categóricas.")

    # Paso extra: guarda csv en 'data/clean_data'
    carpeta_clean_data = "data/clean_data"
    nombre_base = os.path.basename(archivo_actual).replace(".csv", "")
    nombre_limpio = nombre_base.replace("_no_nulls", "").replace("_no_outliers", "").replace("_encoded", "").replace("_scaled", "")
    nombre_final = f"{nombre_limpio}_clean.csv"
    ruta_final_clean = os.path.join(carpeta_clean_data, nombre_final)
    shutil.copy(archivo_actual, ruta_final_clean) # copio el csv a 'data/clean_data'

    # Paso 5: Balanceo de datos, normalización y modelos
    smote = plan.get("aplicar_smote", "no")
    prompt_modeling = f"Divide los datos entre train y test del archivo {archivo_actual}. Gestiona el balanceo de datos con aplicar_smote = '{smote}'. Normaliza los datos siempre. Aplica los modelos y reporta los resultados."

    ejecutar_con_retry("print_response", modeling_agent, prompt_modeling)

if __name__ == "__main__":
    main()
