import numpy as np
import pandas as pd
import pandas_gbq as pbq
from datetime import datetime
from tkds import genq

def cargar_datos(fecha_inicial, proyecto, lay, lay_20, consumo, evaluacion, alertas,
	fecha_final=None, 
	tipo = "alertas"):
	"""
	Función para obtener datos para el análisis de calidad
	"""

	query = genq.genera_queries_calidad(fecha_inicial, 
		fecha_final, 
		lay, 
		lay_20, 
		consumo, 
		evaluacion, 
		alertas,
		tipo)

	df = pbq.read_gbq(query, project_id = proyecto)

	return df

def f_calidad(arr):
    return arr.notna().sum()/arr.size*100

def f_efectividad(arr):
    return arr.sum()/arr.count()*100


def grafica_linea(df, var):

	df_plot = df[[var, 'fecha']].groupby('fecha').mean()

	df_plot[var].plot(ax=ax)
	ax.set_ylabel('% Calidad')
	ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
	ax.set(title = f"Porcentaje de Calidad Diaria")

def grafica_barras(df, var):

	df_plot = df[['fch_registro', var]].\
	groupby(pd.Grouper(key='fch_registro',freq='W')).mean()

	ax.bar(df_plot.index - timedelta(days=6), df_plot[var], width=5, align = 'edge')
	ax.set_ylabel('% Calidad')
	ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(2,32,2)))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
	ax.xaxis.set_minor_locator(mdates.MonthLocator())
	ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
	ax.set(title = f"Porcentaje de Calidad Semanal")
	plt.ylim(bottom=30)


def grafica_linea2(df, df2, var, ax, ax2):

	df_plot = pd.concat([df[[var, 'fch_registro']].groupby('fch_registro').mean(), df2], axis=1)

	df_plot[var].plot(ax=ax)
	df_plot[['efectividad']].plot(ax=ax2, color = 'orange', legend=False)
	ax.set_ylabel('% Calidad')
	ax2.set_ylabel('% Efectividad')
	ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
	handles, labels = ax.get_legend_handles_labels()
	handles2, labels2 = ax2.get_legend_handles_labels()
	ax.set(title = f"Calidad vs Efectividad Diaria")
	fig.legend(handles+handles2, labels+labels2)

def grafica_barras2(df, df2, var, ax, ax2):
	df_plot = pd.concat([df[[var, 'fch_registro']].\
		groupby(pd.Grouper(key='fch_registro',freq='W')).mean(),
		df2.reset_index().groupby(pd.Grouper(key='fch_registro',freq='W')).mean()],
		axis=1)

	ax.bar(df_plot.index - timedelta(days=6), df_plot[var], 
	    width=5,
	    align = 'edge',
	    color = 'tab:blue', 
	    alpha = 0.6
	    )
	ax2.plot(df_plot.index - timedelta(days=3), df_plot['efectividad'], color='tab:orange')
	ax.set_ylabel('% Calidad')
	ax2.set_ylabel('% Efectividad')
	ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(2,32,7)))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
	ax.xaxis.set_minor_locator(mdates.MonthLocator())
	ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
	ax.set_ylim(30, 100)
	ax2.set_ylim(0, 30)