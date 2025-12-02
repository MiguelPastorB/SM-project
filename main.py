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

    # Definimos carpetas
    raw_folder = os.path.join("data", "raw")
    carpeta_processed = os.path.join("data", "processed_data") 
    carpeta_clean_data = os.path.join("data", "clean_data")

    # Definimos rutas y archivos
    try:
        # 1. Intentamos leer la carpeta
        if not os.path.exists(raw_folder):
            raise FileNotFoundError(f"No existe la carpeta {raw_folder}")

        files = os.listdir(raw_folder)
        
        # 2. Si no hay archivos, lanzamos error manual
        if not files:
            raise Exception("La carpeta está vacía. Por favor añade un .csv")
        
        # Listamos los archivos en data/raw
        for file in files:
            print(file)
        archivo_objetivo = files[0]  # tomamos el primer archivo encontrado y devuelve "Bullying1.csv"
        archivo_actual = os.path.join(raw_folder, archivo_objetivo)
        print(f"\n Cogiendo el primer archivo: '{archivo_objetivo}'")
    except Exception as e:
        # Si pasa CUALQUIER cosa mala (no carpeta, vacía, error de permisos...)
        print(f"\n Error crítico seleccionando archivo: {e}")
        print("\n Deteniendo ejecución.")
        return # Cortamos el programa aquí
    
    # para no repetir código en los if
    clean_name = os.path.basename(archivo_objetivo).split("_")[0] if "_" in os.path.basename(archivo_objetivo) else os.path.basename(archivo_objetivo).split(".")[0]
    
    # Paso 1: reporte general
    prompt_quality_report = f"Hazme un reporte de calidad de datos del archivo {archivo_actual}"
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

        nuevo = os.path.join(carpeta_processed, f"{clean_name}_encoded.csv")
        if os.path.exists(nuevo):
            archivo_actual = nuevo
    else:
        print(f"El archivo {archivo_actual} no tiene columnas categóricas.")

    # Paso extra: guarda csv en 'data/clean_data'
    nombre_final = f"{clean_name}_clean.csv"
    ruta_final_clean = os.path.join(carpeta_clean_data, nombre_final)
    shutil.copy(archivo_actual, ruta_final_clean) # copio el csv a 'data/clean_data'

    # Paso 5: Balanceo de datos, normalización y modelos
    smote = plan.get("aplicar_smote", "no")
    prompt_modeling = f"Divide los datos entre train y test del archivo {archivo_actual}. Gestiona el balanceo de datos con aplicar_smote = '{smote}'. Normaliza los datos siempre. Aplica los modelos y reporta los resultados."

    ejecutar_con_retry("print_response", modeling_agent, prompt_modeling)

if __name__ == "__main__":
    main()
