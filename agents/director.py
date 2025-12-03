import os
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class DirectorResponse(BaseModel):
    null_strategy: str = Field(..., description="Estrategia para manejar nulos: 'drop', 'knn', 'skip'")
    outliers_strategy: str = Field(..., description="Estrategia para manejar outliers: 'drop', 'capping', 'skip'")
    encoding_strategy: str = Field(..., description="Estrategia para encoding: 'get_dummies', 'skip'")
    use_smote: str = Field(..., description="Decisión sobre aplicar SMOTE: 'yes', 'no'")


# This agent is the Director who makes strategic decisions based on data quality reports
# Agent
strategy_agent = Agent(
    name="Agente Director",
    model=Gemini(id="gemini-2.5-flash", api_key= os.environ["GOOGLE_API_KEY"]),
    markdown=True,
    output_schema=DirectorResponse, # Using Pydantic model for structured output
    description="Eres el Director de Data Science. Tomas decisiones estratégicas basadas en reportes de calidad.",
    instructions=[
        "Recibirás un reporte de calidad de un dataset y lo mostrarás.",
        "Analiza las dimensiones, nulos, outliers, columnas categóricas ('cols_cat') y desbalanceo.",
        "Debes tomar 5 decisiones basadas en el análisis del reporte.",
        
        "Debes seguir las siguientes estrategias para tomar tu decisión:",
        "1. NULL_STRATEGY:",
        "   - Si nulos < 5%, del total entonces eliges 'drop'.",
        "   - Si nulos >= 5%, entonces eliges 'knn'.",
        "   - Si no hay nulos entonces eliges 'skip'.",
        
        "2. OUTLIERS_STRATEGY:",
        "   - Si filas totales < 1000 entonces eliges 'capping'.",
        "   - Si filas totales >= 1000 entonces eliges 'drop'.",
        "   - Si no hay outliers entonces eliges 'skip'.",
        
        "3. ENCODING_STRATEGY:",
        "   - Si hay columnas categóricas ('cols_cat') aplica 'get_dummies' a las columnas categóricas.",
        "   - Si no hay columnas categóricas aplica 'skip'.",

        "4. SCALING_METHOD: siempre aplica normalización de datos con 'standard'",
        
        "5. USE_SMOTE:",
        "   - Si existe un desbalanceo de datos, entonces eliges 'yes'.",
        "   - Si los datos están balanceados entonces eliges 'no'."
    ]
)