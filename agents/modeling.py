import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend (for environments without display)
import matplotlib.pyplot as plt
import seaborn as sns
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools import tool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@tool
def train_and_test_model(filepath: str, use_smote: str = "no") -> str:
    # Read the CSV file and handle potential errors
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return f"Error: El archivo '{filepath}' no fue encontrado."
    except pd.errors.EmptyDataError:
        return f"Error: El archivo '{filepath}' está vacío."
    except Exception as e:
        return f"Error leyendo el archivo: {e}"

    # train-test split, scaling, SMOTE application, model training and evaluation
    try:
        # We assume the target variable is the last column
        target_col = df.columns[-1]

        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Summary of the process
        process_summary = ""
        process_summary += f"**Dimensiones:** Train={X_train.shape[0]}, Test={X_test.shape[0]}"

        # We scale the features
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply SMOTE if indicated
        if use_smote.lower() == "yes":
            try:
                smote = SMOTE(random_state=42)
                X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
                value_counts_smote = y_train_final.value_counts()
                value_percentages_smote = y_train_final.value_counts(normalize=True) * 100
                process_summary += f"SMOTE aplicado:** La distribución de train es:\n" 
                process_summary += f"Frecuencia:{value_counts_smote}\n"
                process_summary += f"Porcentaje:{value_percentages_smote}"
            except Exception as e:
                return f"Error aplicando SMOTE: {e}"
        # If not applying SMOTE, use the scaled data as is
        elif use_smote.lower() == "no":
            X_train_final, y_train_final = X_train_scaled, y_train
            process_summary += f"SMOTE no aplicado"
        else:
            return "Parámetro 'use_smote' no reconocido. Usa 'yes' o 'no'."

        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_final, y_train_final)

        # Predictions
        y_train_pred = clf.predict(X_train_final)
        y_test_pred = clf.predict(X_test_scaled)

        # Calculate metrics
        calc_metrics = lambda y_true, y_p: {
            'Accuracy': accuracy_score(y_true, y_p),
            'F1-Score': f1_score(y_true, y_p),
            'Recall': recall_score(y_true, y_p),
            'Precision': precision_score(y_true, y_p)
        }
        metrics_train = calc_metrics(y_train_final, y_train_pred)
        metrics_test = calc_metrics(y_test, y_test_pred)

        # Create metrics table
        metrics_table = (
            f"| Métrica | Train Set | Test Set |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **F1-Score** | {metrics_train['F1-Score']:.3f} | {metrics_test['F1-Score']:.3f} |\n"
            f"| **Recall** | {metrics_train['Recall']:.3f} | {metrics_test['Recall']:.3f} |\n"
            f"| **Precision** | {metrics_train['Precision']:.3f} | {metrics_test['Precision']:.3f} |\n"
            f"| **Accuracy** | {metrics_train['Accuracy']:.3f} | {metrics_test['Accuracy']:.3f} |"
        )
        
        # Confusion Matrix plot
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión (Target: {target_col})\nSMOTE: {use_smote}')
        plt.ylabel('Realidad')
        plt.xlabel('Predicción')
        
        # Save the confusion matrix plot
        output_folder = os.path.join("data", "clean_data")
        clean_name = os.path.basename(filepath).split("_")[0] if "_" in os.path.basename(filepath) else os.path.basename(filepath).split(".")[0]
        img_name = f"{clean_name}_confusion_matrix.png"
        path_img = os.path.join(output_folder, img_name)
        plt.savefig(path_img)
        plt.close()
        
        return (
            f"### Rendimiento del Modelo Random Forest\n"
            f"{process_summary}\n\n"
            f"### Comparativa Train vs Test (Detección Overfitting)\n"
            f"{metrics_table}\n\n"
            f"**Visualización:** `{path_img}`"
        )
    except Exception as e:
        return f"Error durante el entrenamiento y evaluación del modelo: {e}"


# Agent
modeling_agent = Agent(
    name="Agente Data Scientist",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[train_and_test_model],
    markdown=True,
    instructions=[
        "Eres un Data Scientist Senior.",
        "Tu objetivo es entrenar y evaluar modelos de machine learning correctamente.",
        "Tu herramienta principal es 'train_and_test_model'.",
        "Recibe el archivo y la decisión de aplicar balanceo de datos con SMOTE o no aplicar balanceo de datos.",
        "Si aplicas SMOTE indica que los datos están balanceados con los porcentajes de cada clase. ",
        "Genera un análisis de las métricas obtenidas comparando con train y test para ver si hay overfitting y concluyendo si el modelo predice bien o no. "
    ]
)