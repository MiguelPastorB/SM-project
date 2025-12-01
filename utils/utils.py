import time
import os
import shutil

def ejecutar_con_retry(function, agente, prompt, intentos=3, espera_inicial=5):
    for i in range(intentos):
        try:
            if function == 'run':
                return agente.run(prompt, stream=False)
            elif function == 'print_response':
                return agente.print_response(prompt, stream=True)
            else:
                raise Exception
        
        except Exception as e:
            msg = str(e)
            # Detectamos errores típicos de API saturada
            if "503" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg or "UNAVAILABLE" in msg:
                print(f"⚠️ Error de conexión (Intento {i+1}/{intentos}): {msg}")
                
                if i < intentos - 1: # Si no es el último intento, esperamos
                    print(f"⏳ Esperando {espera_inicial} segundos para reintentar...")
                    time.sleep(espera_inicial)
                    espera_inicial *= 2 # Duplicamos el tiempo de espera (5s -> 10s -> 20s)
                else:
                    print("❌ Se agotaron los reintentos.")
                    raise e # Si falló la última vez, lanzamos el error real
            else:
                # Si es un error de código fallamos inmediatamente
                raise e
            
def limpiar_datos_antiguos():
    carpetas_a_limpiar = ["data/processed_data", "data/clean_data"]
    
    for carpeta in carpetas_a_limpiar:
        if os.path.exists(carpeta):
            # shutil.rmtree borra la carpeta y todo su contenido recursivamente
            shutil.rmtree(carpeta)
        
        # Volvemos a crear la carpeta vacía
        os.makedirs(carpeta)