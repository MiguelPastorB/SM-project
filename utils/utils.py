import time
import os
import shutil

def retry(function, agent, prompt, attempts=3, initial_wait=5):
    """
    Executes a function on an agent with retry logic for connection-related errors.

    Args:
        function_name (str): The function to call ('run' or 'print_response').
        agent (object): The agent object with the method to call.
        prompt (str): The prompt string to send to the agent.
        attempts (int): Number of retry attempts (default 3).
        initial_wait (int): Initial wait time between retries in seconds (default 5).

    Returns:
        The result of the agent method call.

    Raises:
        Exception: If retries are exhausted or other errors occur.
    """
    for i in range(attempts):
        try:
            if function == 'run':
                return agent.run(prompt, stream=False) # Run without streaming
            elif function == 'print_response':
                return agent.print_response(prompt, stream=True) # Print with streaming
            else:
                raise Exception
        
        except Exception as e:
            msg = str(e)
            # Check for connection-related errors
            if "503" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg or "UNAVAILABLE" in msg:
                print(f"Error de conexión (Intento {i+1}/{attempts}): {msg}")
                if i < attempts - 1:
                    print(f"Esperando {initial_wait} segundos para reintentar...")
                    time.sleep(initial_wait)
                    initial_wait *= 2 # Exponential backoff
                else:
                    print("Se agotaron los reintentos.")
                    raise e # Rethrow exception after final attempt
            else:
                raise e # Rethrow non-connection-related exceptions
            
def clear_old_data():
    """
    Deletes and recreates directories for processed and cleaned data.

    Directories:
        - data/processed_data
        - data/clean_data

    Ensures old processed data is removed before new processing.
    """
    folders_to_clear  = ["data/processed_data", "data/clean_data"] # Define folders to clear
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder) # Remove existing folder
        os.makedirs(folder) # Create empty folder