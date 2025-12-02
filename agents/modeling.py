import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') # esto es para que no de error en entornos sin interfaz gráfica y solo muestre las imágenes guardadas jpg/png
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

# Cargamos las claves
load_dotenv()

# Esta herramienta separa los datos en train y test y si es necesario aplica balanceo de datos. Luego aplica los modelos
# Guarda el resultado en 'data/processed_data'
@tool
def entrenar_y_evaluar(filepath: str, aplicar_smote: str = "no") -> str:

    df = pd.read_csv(filepath)

    # Seleccionamos la variable objetivo como la última columna del dataset
    target_col = df.columns[-1]

    # Separamos X e y
    X = df.drop(columns=[target_col]) # target_col es la variable objetivo
    y = df[target_col]

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    info_proceso = []
    info_proceso.append(f"**Dimensiones:** Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    # Normalizamos los datos
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Diferenciar entre aplicar smote o no según lo decida el agente director
    if aplicar_smote.lower() == "si":
        smote = SMOTE(random_state=42)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        value_counts_smote = y_train_final.value_counts()
        value_percentages_smote = y_train_final.value_counts(normalize=True) * 100
        info_proceso.append(f"SMOTE aplicado:** La distribución de train es:\n" 
                            f"Frecuencia:{value_counts_smote}\n"
                            f"Porcentaje:{value_percentages_smote}"
                            )
    elif aplicar_smote.lower() == "no":
        X_train_final, y_train_final = X_train, y_train
        info_proceso.append(f"SMOTE no aplicado")

    # Entrenar Modelo Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_final, y_train_final)

    # Predecir sobre train(para las métricas) y test
    y_train_pred = clf.predict(X_train_final)
    y_test_pred = clf.predict(X_test_scaled)

    # Métricas
    calc_metrics = lambda y_true, y_p: {
        'Accuracy': accuracy_score(y_true, y_p),
        'F1-Score': f1_score(y_true, y_p),
        'Recall': recall_score(y_true, y_p),
        'Precision': precision_score(y_true, y_p)
    }
    metrics_train = calc_metrics(y_train_final, y_train_pred)
    metrics_test = calc_metrics(y_test, y_test_pred)

    tabla_comparativa = (
        f"| Métrica | Train Set | Test Set |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **F1-Score** | {metrics_train['F1-Score']:.3f} | {metrics_test['F1-Score']:.3f} |\n"
        f"| **Recall** | {metrics_train['Recall']:.3f} | {metrics_test['Recall']:.3f} |\n"
        f"| **Precision** | {metrics_train['Precision']:.3f} | {metrics_test['Precision']:.3f} |\n"
        f"| **Accuracy** | {metrics_train['Accuracy']:.3f} | {metrics_test['Accuracy']:.3f} |"
    )
    
    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión (Target: {target_col})\nSMOTE: {aplicar_smote}')
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    
    # Guardar imagen
    output_folder = "data/processed_data"
    clean_name = os.path.basename(filepath).split("_")[0] if "_" in os.path.basename(filepath) else os.path.basename(filepath).split(".")[0]
    img_name = f"{clean_name}_confusion_matrix.png"
    path_img = os.path.join(output_folder, img_name)
    plt.savefig(path_img)
    plt.close()
    
    return (
        f"### Rendimiento del Modelo Random Forest\n"
        f"{chr(10).join(info_proceso)}\n\n"
        f"### Comparativa Train vs Test (Detección Overfitting)\n"
        f"{tabla_comparativa}\n\n"
        f"**Visualización:** `{path_img}`"
    )


# Agente
modeling_agent = Agent(
    name="Agente Data Scientist",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    tools=[entrenar_y_evaluar],
    markdown=True,
    instructions=[
        "Eres un Data Scientist Senior.",
        "Tu objetivo es entrenar y evaluar modelos de machine learning correctamente.",
        "Recibe el archivo y la decisión de aplicar balanceo de datos con SMOTE o no aplicar balanceo de datos.",
        "Si aplicas SMOTE indica que los datos están balanceados con los porcentajes de cada clase. ",
        "Genera un análisis de las métricas obtenidas comparando con train y test para ver si hay overfitting y concluyendo si el modelo predice bien o no. "
    ]
)