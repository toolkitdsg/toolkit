import numpy as np
import pandas as pd
import pandas_gbq as pbq
from datetime import datetime
from tkds import genq

def cargar_datos(fecha_inicial, fecha_final=None, tipo = "alertas", proyecto = "fugasfraudesgmma-pro"):
	"""
	Función para obtener datos para el análisis de calidad
	"""

	query = genq.genera_queries_calidad(fecha_inicial, fecha_final, tipo)

	df = pbq.read_gbq(query, project_id = proyecto)

	return df

def f_calidad(arr):
    return arr.notna().sum()/arr.size*100

def f_efectividad(arr):
    return arr.sum()/arr.count()*100