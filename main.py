# ============================================
# API DE PREDICCIÓN DE RECAUDACIÓN TRIBUTARIA
# ============================================

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================
# 1. CARGAR MODELO
# ============================================

modelo = joblib.load("modelo_random_forest.joblib")


# ============================================
# 2. CREAR API
# ============================================

app = FastAPI(
    title="API de Predicción de Recaudación Tributaria",
    description="API para predecir el valor de recaudación tributaria mediante Machine Learning.",
    version="1.0.0"
)


# ============================================
# 3. CONFIGURAR HTML
# ============================================

templates = Jinja2Templates(directory="templates")


# ============================================
# 4. DATOS DE ENTRADA
# ============================================

class DatosPrediccion(BaseModel):

    ANIO: int
    MES_NUM: int
    GRUPO_IMPUESTO: str
    SUBGRUPO_IMPUESTO: str
    IMPUESTO: str
    GRAN_CONTRIBUYENTE: str
    CODIGO_OPERA_FAMILIA: str
    TIPO_CONTRIBUYENTE: str
    PROVINCIA: str
    CANTON: str


# ============================================
# 5. MOSTRAR FORMULARIO
# ============================================

@app.get("/", response_class=HTMLResponse)
def inicio(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )


# ============================================
# 6. REALIZAR PREDICCIÓN
# ============================================

@app.post("/predict")
def predecir(datos: DatosPrediccion):

    # Convertir los datos recibidos a diccionario
    datos_dict = datos.model_dump()

    # Crear DataFrame
    entrada = pd.DataFrame([datos_dict])

    # Realizar predicción
    prediccion = modelo.predict(entrada)

    # Obtener resultado
    valor_predicho = float(prediccion[0])

    return {
        "valor_recaudado_predicho": valor_predicho
    }


# ============================================
# 7. VERIFICAR API
# ============================================

@app.get("/api")
def verificar_api():

    return {
        "mensaje": "API funcionando correctamente",
        "modelo": "Random Forest Regressor",
        "estado": "Activo"
    }