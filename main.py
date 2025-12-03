import os
import shutil
import json
from agents.quality import quality_agent
from agents.nan_imputer import nan_imputer_agent
from agents.outliers import outlier_agent
from agents.one_hot import one_hot_agent
from agents.modeling import modeling_agent
from agents.director import strategy_agent
from utils.utils import retry
from utils.utils import clear_old_data
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load prompts from JSON file
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "config", "prompts.json")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

# Main execution function
def main():
    # Remove old processed files
    clear_old_data()

    # Define main directories
    raw_folder = os.path.join("data", "raw")
    folder_processed = os.path.join("data", "processed_data") 
    folder_clean_data = os.path.join("data", "clean_data")

    try:
        # Check if raw data folder exists
        if not os.path.exists(raw_folder):
            raise FileNotFoundError(f"No existe la carpeta {raw_folder}")
        # List files inside raw folder
        files = os.listdir(raw_folder)
        
        # Stop if no CSV available
        if not files:
            raise Exception("La carpeta está vacía. Por favor añade un .csv")
        
        # Print all found files for visibility
        for file in files:
            print(file)
        target_file = files[0]
        current_file = os.path.join(raw_folder, target_file)
        print(f"\n Cogiendo el primer archivo: '{target_file}'")
    except Exception as e:
        # Critical error: stop execution
        print(f"\n Error crítico seleccionando archivo: {e}")
        print("\n Deteniendo ejecución.")
        return
    
    # Extract clean base name without prefix or file extension
    clean_name = os.path.basename(target_file).split("_")[0] if "_" in os.path.basename(target_file) else os.path.basename(target_file).split(".")[0]
    
    # Step 1: Data Quality Report
    prompt_quality_report = PROMPTS["quality_report"].format(filename=current_file)
    quality_report = retry("run", quality_agent, prompt_quality_report) # Get quality report
    text_report = quality_report.content # Extract text content
    print(text_report)

    # Step 2: Strategy Planning
    prompt_director = PROMPTS["director"].format(report=text_report)
    report_director = retry("run", strategy_agent, prompt_director) # Get strategy plan
    plan = dict(report_director.content) # Convert to dictionary

    # Step 3: Null Value Handling
    action = plan.get("null_strategy", "skip")
    prompt_nan = PROMPTS["nan"].format(filename=current_file, action=action)
    if action != "skip":
        retry("print_response", nan_imputer_agent, prompt_nan) # Execute imputation
        # Update dataset path if new file was created
        new = os.path.join(folder_processed, f"{clean_name}_no_nulls.csv")
        if os.path.exists(new):
            current_file = new 
    else:
        print(f"El archivo {current_file} no tiene valores nulos.")

    # Step 4: Outlier Handling
    action = plan.get("outliers_strategy", "skip")
    prompt_outlier = PROMPTS["outliers"].format(filename=current_file, action=action)
    if action != "skip":
        retry("print_response", outlier_agent, prompt_outlier) # Execute outlier handling
        # Update dataset path if new file was created
        new = os.path.join(folder_processed, f"{clean_name}_no_outliers.csv")
        if os.path.exists(new):
            current_file = new
    else:
        print(f"El archivo {current_file} no tiene outliers.")

    # Step 5: One-Hot Encoding for Categorical Variables
    action = plan.get("encoding_strategy", "skip")
    prompt_one_hot = PROMPTS["one_hot"].format(filename=current_file)
    if action == "get_dummies":
        retry("print_response", one_hot_agent, prompt_one_hot) # Execute one-hot encoding
        # Update dataset path if new file was created
        new = os.path.join(folder_processed, f"{clean_name}_encoded.csv")
        if os.path.exists(new):
            current_file = new
    else:
        print(f"El archivo {current_file} no tiene columnas categóricas.")

    # Step 6: Final Clean Data Copy
    final_name = f"{clean_name}_clean.csv"
    path_final_clean = os.path.join(folder_clean_data, final_name)
    shutil.copy(current_file, path_final_clean) # Copy final cleaned file

    # Step 7: Modeling
    smote = plan.get("use_smote", "no") # Check SMOTE decision
    prompt_modeling = PROMPTS["modeling"].format(filename=current_file, plan=smote)

    retry("print_response", modeling_agent, prompt_modeling) # Run modeling agent

if __name__ == "__main__":
    main()
