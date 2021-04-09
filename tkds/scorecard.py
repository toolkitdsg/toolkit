import numpy as np
import pandas as pd
import pandas_gbq as pbq
from tkds import genq

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import seabron as sns


def tablas_alertas(fecha_inicial, fecha_final, 
  lay="fugasfraudesgmma-qa.layout_hi.GMM_LAYH_2021_TEMP", 
  project_id="fugasfraudesgmma-qa"):
  
  # Se obtiene el query
  query = genq.query_alertas(fecha_inicial, fecha_final, lay)

  # Se carga la tabla
  df = pbq.read_gbq(query, project_id = project_id)

  # Se agregan columnas
  df['score_1'] = df.score_1.astype('float64')

  f = lambda x: 'menor 0.6' if x<0.6 else  'mayor 0.6'
  df['umbral'] = df['score_1'].map(f)

  df['recalibracion'] = df.model_name.str.extractall(r"(recal[12])+").unstack().\
  fillna('').agg(' '.join, axis=1).replace('recal1 recal1', 'recal1').replace('recal1 ', 'recal1')

  df['fch_asignacion'] = df.fch_asignacion.dt.date

  df['modelo'] = df.model_name.str.replace(r"(_recal[12])+","").str.replace("-","")

  # Se crea la tabla pivote
  df_wide = df.melt(['id_registro',
                   'fch_registro',
                   'fch_asignacion',
                   'fch_revision',
                   'score_qa',
                   'score_pro',
                   'score_1', 
                   'estatus',
                   'patron',
                   'model_name',
                   'ambiente',
                   'umbral',
                   'recalibracion',
                   'modelo'])
  df_wide['modelo_sin_rec'] = df_wide.value.str.replace(r"(_recal[12])+","")
  df_wide = df_wide.rename(columns={'modelo': 'modelos', 'value': 'modelo'})

  return df, df_wide


def umbral_optimo_ponderado(df, modelo, w=1):

	## Diccionario para las etiquetas
	dic_estatus = {'SIN DESVIACIÓN': 0, 'CON DESVIACIÓN': 1}

	## Filtro para el modelo correspondiente
	df_plot = df[df['modelo']==modelo]

	## Cálculo de presision, recall y f1
	precision, recall, thresholds = precision_recall_curve(
		df_plot.estatus.map(dic_estatus), 
    	df_plot.score_1)
	f1 = (w+1)*(precision**w)*(recall)/((precision**w)+w*recall*(precision**(w-1)))

	## Métricas para 0.5
	r = np.min(abs(thresholds-0.5))
	i = np.where(((0.5 - r) <= thresholds) & (thresholds <= (0.5 + r)))
	f1_50 = float(f1[i])
	precision_50 = float(precision[i])
	recall_50 = float(recall[i])


	## Métricas umbral óptimo
	f1_max = np.max(f1)
	i_max = np.where(f1==f1_max)
	umbral_max = float(thresholds[i_max])
	precision_max = float(precision[i_max])
	recall_max = float(recall[i_max])

	## Gráficas
	fig, ax = plt.subplots(1,3)

	sns.stripplot(x="estatus", y="score_1", data=df_plot,
		order = ['CON DESVIACIÓN', 'SIN DESVIACIÓN'],
		palette = ("g","r"),
		size=1,
		alpha=0.5,
		ax = ax[0])
	sns.boxplot(x="estatus", y="score_1", data=df_plot,
		order = ['CON DESVIACIÓN', 'SIN DESVIACIÓN'],
		palette = ("g","r"),
		width=0.9,
		linewidth=0.6,
		boxprops=dict(alpha=.3),
		showfliers = False,
		ax = ax[0])
	ax[0].axhline(thresholds[i_max],0,1, c='k', ls='--', lw=1, label="umbral óptimo")
	ax[0].set_xlabel('')

	ax[1].plot(precision, recall, label="precision recall curve")
	ax[1].plot(precision[i_max], recall[i_max], 'o', markersize=8, fillstyle="none", c='k')
	ax[1].set_xlabel('Precision')
	ax[1].set_ylabel('Recall')

	ax[2].plot(thresholds, precision[:-1], label="precision")
	ax[2].plot(thresholds, recall[:-1], label="recall")
	ax[2].plot(thresholds, f1[:-1], label="f1")
	ax[2].axvline(thresholds[i_max],0,1, c='k', ls='--', lw=1, label="umbral óptimo")
	ax[2].set_xlabel('Umbral')

	plt.legend(loc="best")
	plt.suptitle(f"Análisis Precisión vs Umbral {modelo}")
	plt.show()

	print(f"La precisión con el umbral de 0.5 es {round(precision_50,2)}")
	print(f"El recall con el umbral de 0.5 es {round(recall_50,2)}")
	print(f"El valor de f1 con el umbral de 0.5 es {round(f1_50,2)}")
	print("\n")
	print(f"El umbral óptimo es {round(umbral_max,2)}")
	print(f"La precisión con el umbral óptimo es {round(precision_max,2)}")
	print(f"El recall con el umbral óptimo es {round(recall_max,2)}")
	print(f"El valor de f1 con el umbral óptimo es {round(f1_max,2)}")

	return precision_50, recall_50, f1_50, umbral_max, precision_max, recall_max, f1_max

def color_min(val, l1, l2, pts = False):
    p=0
    if val < l1:
        color = 'lightcoral'
        p = 2
    elif val < l2:
        color = 'lightyellow' 
        p = 1
    elif pd.isnull(val):
        color = 'lightyellow' 
        p = 1
    else:
        color = 'lightgreen'
        
    r = f'background-color: {color}'
    
    return p if pts is True else r

def color_max(val, l1, l2, pts = False):
    p=0
    if val > l1:
        color = 'lightcoral'
        p = 2
    elif val > l2:
        color = 'lightyellow' 
        p = 1
    else:
        color = 'lightgreen'
    
    r = f'background-color: {color}'
    
    return p if pts is True else r

def color_glob(val):
    if val > 2:
        color = 'red'
    elif val > 1:
        color = 'yellow' 
    else:
        color = 'green'
    
    r = f'background-color: {color}'
    
    return r

def tabla_scorecard(df):

	df['Global'] = df['Train'].apply(color_min, l1=0.7, l2=0.85, pts =True) +\
	df['Train-Test'].apply(color_max, l1=0.25, l2=0.15, pts =True) +\
	df['Real_3sem'].apply(color_min, l1=0, l2=0.1, pts =True) +\
	df['Real_6sem'].apply(color_min, l1=0.05, l2=0.15, pts =True)

	sty = df.style.\
	applymap(color_min, subset='Train',l1=0.7, l2=0.85).\
	applymap(color_max, subset='Train-Test',l1=0.25, l2=0.15).\
	applymap(color_min, subset='Real_3sem',l1=0, l2=0.1).\
	applymap(color_min, subset='Real_6sem',l1=0.05, l2=0.15).\
	applymap(color_glob, subset='Global').\
	set_properties(**{'border':'2px solid white', 'width':'75px'}).\
	format({"Train": "{:.1%}", 
		"Train-Test": "{:.1%}", 
		"Real_3sem": "{:.1%}", 
		"Alertas_3sem": "{:^.0f}",
		"Real_6sem": "{:.1%}",
		"Alertas_6sem": "{:^.0f}",
		"Global":""})

	return sty

def score_diario_modelo(df, filtro):
  df_plot=df.loc[df['modelo']==filtro,['mes', 'score_1', 'estatus']].sort_values('mes')

  fig, ax = plt.subplots()
  ax=sns.stripplot(x="mes", y="score_1", hue = 'estatus', 
                   data=df_plot,
                   size=3,
                   alpha=0.6,
                   palette = ("g","r"))
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
  ax.set(title = f"Distribución de score para los desvíos mensuales del modelo {filtro}")
  plt.show()


def grafica_efectividad_modelo(df, modelo, ax):
  df_plot = df[df['modelo']==modelo].groupby(['estatus', 'fch_asignacion']).count().\
          id_registro.unstack(0).reset_index().fillna(0)
  df_plot['alertas'] = df_plot.get('CON DESVIACIÓN',0)+df_plot.get('EN REVISIÓN',0)+df_plot.get('SIN DESVIACIÓN',0)
  df_plot['precision'] = df_plot.get('CON DESVIACIÓN',0)/df_plot['alertas']*100
  df_plot = df_plot.sort_values(['fch_asignacion'])

  ax.bar(df_plot.fch_asignacion, df_plot.get('CON DESVIACIÓN',0)/df_plot['alertas']*100, 0.5,
        color = "g", label='Con Desviación')
  ax.bar(df_plot.fch_asignacion, df_plot.get('EN REVISIÓN',0)/df_plot['alertas']*100, 0.5,
        bottom = df_plot.get('CON DESVIACIÓN',0)/df_plot['alertas']*100, 
        color = "y", label='En Revisión')
  ax.bar(df_plot.fch_asignacion, df_plot.get('SIN DESVIACIÓN',0)/df_plot['alertas']*100, 0.5, 
        bottom = (df_plot.get('CON DESVIACIÓN',0) + df_plot.get('EN REVISIÓN',0))/df_plot['alertas']*100, 
        color = "r", label='Sin Desviación')
  for i,j in zip(df_plot.fch_asignacion,df_plot.precision):
      ax.annotate(str(int(round(j,0)))+'%',
                  xy=(i,j),
                  ha='center', 
                  va='bottom',
                  size = 9)
  ax2 = ax.twinx()
  ax2.plot(df_plot.fch_asignacion, df_plot['alertas'], color = "b", label='Alertas')
  #ax.xaxis.set_major_locator(mticker.FixedLocator(df_plot['fch_asignacion'].tolist()))
  #ax.set_xticklabels(pd.to_datetime(df_plot['fch_asignacion']).dt.strftime("%d-%b"))
  ax2.grid(False)
  ax.set_ylabel(f"{modelo}")

def grafica_efectividad(df, ax, filtro=None, train = True):

  if filtro is not None:
    df  = df[df['train_test']==filtro]
    print("Filtro activado")

  if train is True:
    df_plot = df[df['score_1']>0.5].groupby(['estatus', 'modelo']).count().\
              id_registro.unstack(0).fillna(0)
    df_plot['alertas'] = df_plot['CON DESVIACIÓN']+df_plot['SIN DESVIACIÓN']
    df_plot['precision'] = df_plot['CON DESVIACIÓN']/df_plot['alertas'] * 100
    df_plot = df_plot.sort_values(['modelo'])

    ax.bar(df_plot.index, df_plot['CON DESVIACIÓN']/df_plot['alertas']* 100, 0.5,
      color = "g", label='Con Desviación')
    ax.bar(df_plot.index, df_plot['SIN DESVIACIÓN']/df_plot['alertas']* 100, 0.5, 
      bottom = df_plot['CON DESVIACIÓN']/df_plot['alertas']*100, 
      color = "r", label='Sin Desviación')

  elif train is False:
    df_plot = df[pd.notna(df.modelo)].groupby(['estatus', 'modelo']).count().\
          id_registro.unstack(0).fillna(0)
    df_plot['EN REVISIÓN'] = df_plot.get('EN REVISIÓN',0)
    df_plot['alertas'] = df_plot['CON DESVIACIÓN']+df_plot['EN REVISIÓN']+df_plot['SIN DESVIACIÓN']
    df_plot['precision'] = df_plot['CON DESVIACIÓN']/df_plot['alertas'] * 100
    df_plot = df_plot.sort_values(['modelo'])

    ax.bar(df_plot.index, df_plot['CON DESVIACIÓN']/df_plot['alertas'] * 100, 0.5,
      color = "g", label='Con Desviación')
    ax.bar(df_plot.index, df_plot['EN REVISIÓN']/df_plot['alertas'] * 100, 0.5,
      bottom = df_plot['CON DESVIACIÓN']/df_plot['alertas'] * 100, 
      color = "y", label='En Revisión')
    ax.bar(df_plot.index, df_plot['SIN DESVIACIÓN']/df_plot['alertas']*100, 0.5,
      bottom = [i+j for i,j in zip(df_plot['CON DESVIACIÓN'], df_plot['EN REVISIÓN'])]/df_plot['alertas']*100, 
      color = "r", label='Sin Desviación')


  for i,j in zip(df_plot.index,df_plot.precision):
      ax.annotate(str(round(j,1))+'%',
                  xy=(i,j),
                  ha='center', 
                  va='baseline',
                  weight='bold')
  ax2 = ax.twinx()
  ax2.plot(df_plot.index, df_plot['alertas'], color = "b", label='Alertas')
  #ax.legend()
  ax.set_xticklabels(df_plot.index, rotation=30)
  ax.set(title = f"Estatus porcentual por modelo y volumen de alertas")
  ax.set_ylabel(f"Estatus porcentual {filtro}")
  ax2.grid(False)

  return df_plot.precision, df_plot.alertas


def grafica_escenarios(df, ax, ax2):
  df_sim = df[pd.notna(df.modelo)].groupby(['estatus', 'modelo']).count().\
          id_registro.unstack(0).reset_index().fillna(0)
  df_sim['alertas'] = df_sim['CON DESVIACIÓN']+df_sim['EN REVISIÓN']+df_sim['SIN DESVIACIÓN']
  df_sim['precision'] = df_sim['CON DESVIACIÓN']/df_sim['alertas'] * 100
  df_sim = df_sim.sort_values(['precision'], ascending=False)
  df_sim['Con Desviación'] = df_sim['CON DESVIACIÓN'].cumsum()
  df_sim['En Revisión'] = df_sim['EN REVISIÓN'].cumsum()
  df_sim['Sin Desviación'] = df_sim['SIN DESVIACIÓN'].cumsum()
  df_sim['Alertas'] = df_sim['alertas'].cumsum()
  df_sim['Efectividad'] = df_sim['Con Desviación']/df_sim['Alertas']*100
  df_plot= df_sim[['modelo', 'Con Desviación', 'En Revisión', 'Sin Desviación', 'Alertas', 'Efectividad']].reset_index(drop=True)
  df_plot['modelo eliminado'] = df_plot.iloc[1:,0].append(pd.Series(['Total']), ignore_index=True)
  #df_plot = df_plot.iloc[1:,0].append(pd.Series(['Total']), ignore_index=True)
    
  ax.barh(df_plot['modelo eliminado'], df_plot['Con Desviación'], 0.5, color="g", label='Con Desviación')
  ax.barh(df_plot['modelo eliminado'], df_plot['En Revisión'], 0.5, 
    left = df_plot['Con Desviación'], color = "y", label='En Revisión')
  ax.barh(df_plot['modelo eliminado'], df_plot['Sin Desviación'], 0.5,
    left = [i+j for i,j in zip(df_plot['Con Desviación'], df_plot['En Revisión'])], 
    color = "r", label='Sin Desviación')

  ax2.plot(df_plot['Efectividad'], df_plot['modelo eliminado'], color = "darkgreen", label='Efectividad', 
           alpha = 0.6)
  # ax.set(title = f"Estatus porcentual por modelo y volumen de alertas de {fecha_inicial} a {fecha_final}")
  ax2.grid(False)

  for i,j in zip(df_plot['modelo eliminado'],df_plot['Alertas']):
      ax.annotate(str(int(j)),
                  xy=(j,i),
                  ha='left', 
                  va='center',
                  fontweight = 'semibold')
      
  for i,j in zip(df_plot['modelo eliminado'],df_plot['Efectividad']):
      ax2.annotate(str(int(j))+'%',
                  xy=(j,i),
                  ha='center', 
                  va='center',
                  color = 'darkgreen',
                  fontweight = 'semibold')
  ax.set_xlabel('Alertas')
  ax2.set_xlabel('% Efectividad')
  ax.set_ylabel('Modelos descartados')


def grafica_distribucion(df, ax):
  df_plot = df.loc[:, ['modelo', 'estatus', 'score_1']].sort_values(['estatus'])

  ax=sns.stripplot(x="modelo", y="score_1", hue="estatus", data=df_plot, 
                   order = df_plot.groupby('modelo').count().sort_values('estatus', ascending=False).index,
                   dodge=True,
                   palette = ("g","r"),
                   size=1,
                   alpha=0.5)
  ax=sns.boxplot(x="modelo", y="score_1", hue="estatus", data=df_plot,
                 order = df_plot.groupby('modelo').count().sort_values('estatus', ascending=False).index,
                 dodge=True,
                 palette = ("g","r"),
                 width=0.9,
                 linewidth=0.6,
                 whis=10,
                 boxprops=dict(alpha=.3),
                 showfliers = False)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
  ax.set(title = f"Distribución de score por estatus y modelo")