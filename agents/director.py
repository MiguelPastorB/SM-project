import os
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Cargamos las claves
load_dotenv()

class DirectorResponse(BaseModel):
    estrategia_nulos: str = Field(..., description="Estrategia para manejar nulos: 'eliminar', 'knn', 'saltar'")
    estrategia_outliers: str = Field(..., description="Estrategia para manejar outliers: 'eliminar', 'capping', 'saltar'")
    estrategia_encoding: str = Field(..., description="Estrategia para encoding: 'get_dummies', 'saltar'")
    aplicar_smote: str = Field(..., description="Decisión sobre aplicar SMOTE: 'si', 'no'")


# La idea es que el agente director tome todas las decisiones siguiendo la lógica de un científico de datos y hable con el resto de agentes mediante JSON
# Agente
strategy_agent = Agent(
    name="Agente Director",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    markdown=True,
    output_schema=DirectorResponse, # configuramos el esquema de salida para que devuelva JSON
    description="Eres el Director de Data Science. Tomas decisiones estratégicas basadas en reportes de calidad.",
    instructions=[
        "Recibirás un reporte de calidad de un dataset y lo mostrarás.",
        "Analiza las dimensiones, nulos, outliers, columnas categóricas ('cols_cat') y desbalanceo.",
        "Debes tomar 5 decisiones basadas en el análisis del reporte.",
        
        "Debes seguir las siguientes estrategias para tomar tu decisión:",
        "1. ESTRATEGIA_NULOS:",
        "   - Si nulos < 5%, del total entonces eliges 'eliminar'.",
        "   - Si nulos >= 5%, entonces eliges 'knn'.",
        "   - Si no hay nulos entonces eliges 'saltar'.",
        
        "2. ESTRATEGIA_OUTLIERS:",
        "   - Si filas totales < 1000 entonces eliges 'capping'.",
        "   - Si filas totales >= 1000 entonces eliges 'eliminar'.",
        "   - Si no hay outliers entonces eliges 'saltar'.",
        
        "3. ESTRATEGIA_ENCODING:",
        "   - Si hay columnas categóricas ('cols_cat') aplica 'get_dummies' a las columnas categóricas.",
        "   - Si no hay columnas categóricas aplica 'saltar'.",

        "4. METODO_ESCALADO: siempre aplica normalización de datos con 'standard'",
        
        "5. APLICAR_SMOTE:",
        "   - Si existe un desbalanceo de datos, entonces eliges 'si'.",
        "   - Si los datos están balanceados entonces eliges 'no'."
    ]
)