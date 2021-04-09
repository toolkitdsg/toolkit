import numpy as np
import pandas as pd
from datetime import datetime

def genera_queries_calidad(fecha_inicial, fecha_final, lay, lay_20, consumo, evaluacion, alertas, tipo):

  if fecha_final is None:
    fecha_final = datetime.today().strftime("%Y-%m-%d")

  print("Datos a cargar del ", fecha_inicial, " al ", fecha_final)

  if tipo == "total":
    query = f"""
    WITH CONSUMO AS(
    SELECT DISTINCT *
    FROM `{consumo}`
    WHERE fch_registro  BETWEEN '{fecha_inicial}' AND '{fecha_final}'
    ),
    MOD4_1 AS (
      SELECT id_registro, fch_registro, AVG(score_1) as score_mod4_v1
      FROM `{evaluacion}`
      WHERE model_name = 'modelo4_v3_recal1'
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      ),
    MOD4_2 AS (
      SELECT DISTINCT id_registro, fch_registro, avg(score_1) as score_mod4_v2
      FROM `{evaluacion}`
      WHERE model_name = 'modelo4_v3_recal2'
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      ),
    PADRE_1 AS (
      SELECT DISTINCT id_registro, fch_registro, avg(score_1) as score_padre_v1
      FROM `{evaluacion}`
      WHERE model_name = 'modelo_padre_recal1'
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      ),
    PADRE_2 AS (
      SELECT DISTINCT id_registro, fch_registro, avg(score_1) as score_padre_v2
      FROM `{evaluacion}`
      WHERE model_name = 'modelo_padre_recal2'
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      ),
    PATRON_1 AS (
      SELECT DISTINCT id_registro, fch_registro, avg(score_1) as score_patron_v1
      FROM `{evaluacion}`
      WHERE model_name = 'modelo_patrones_recal1'
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      ),
    PATRON_2 AS (
      SELECT DISTINCT id_registro, fch_registro, avg(score_1) as score_patron_v2
      FROM `{evaluacion}`
      WHERE model_name = 'modelo_patrones_recal2'
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      ),
    ICD_1 AS (
      SELECT DISTINCT id_registro, fch_registro, avg(score_1) as score_icd_v1
      FROM `{evaluacion}`
      WHERE model_name = 'modelo_sin_icd_excluidos_recal1'
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      ),
    ICD_2 AS (
      SELECT DISTINCT id_registro, fch_registro, avg(score_1) as score_icd_v2
      FROM `{evaluacion}`
      WHERE model_name = 'modelo_sin_icd_excluidos_recal2' 
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      ),
    INV_1 AS (
      SELECT DISTINCT id_registro, fch_registro, avg(score_1) as score_inv_v1
      FROM `{evaluacion}`
      WHERE model_name = 'modelo_sin_invalidos_recal1'
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      ),
    INV_2 AS (
      SELECT DISTINCT id_registro, fch_registro, avg(score_1) as score_inv_v2
      FROM `{evaluacion}`
      WHERE model_name = 'modelo_sin_invalidos_recal2'
      AND fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_registro
      )
    SELECT *
    FROM CONSUMO
    LEFT JOIN MOD4_1 USING (id_registro, fch_registro)
    LEFT JOIN MOD4_2 USING (id_registro, fch_registro)
    LEFT JOIN PADRE_1 USING (id_registro, fch_registro)
    LEFT JOIN PADRE_2 USING (id_registro, fch_registro)
    LEFT JOIN PATRON_1 USING (id_registro, fch_registro)
    LEFT JOIN PATRON_2 USING (id_registro, fch_registro)
    LEFT JOIN ICD_1 USING (id_registro, fch_registro)
    LEFT JOIN ICD_2 USING (id_registro, fch_registro)
    LEFT JOIN INV_1 USING (id_registro, fch_registro)
    LEFT JOIN INV_2 USING (id_registro, fch_registro)
    """

  elif tipo == "alertas":
    query = f"""
    WITH ALERTAS AS (
      SELECT id_registro, fch_registro, CAST(fecha_ejecucion AS DATE) AS fecha_ejecucion, model_name, 
      AVG(score_1) as score_alerta
      FROM `{alertas}`
      WHERE fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      GROUP BY id_registro, fch_Registro, fecha_ejecucion, model_name
      ),
    CONSUMO AS(
      SELECT *, row_number() over (partition by id_registro, fch_registro) as num
      FROM `{consumo}`
      WHERE fch_registro BETWEEN '{fecha_inicial}' AND '{fecha_final}'
      ),
    LAY AS (
      SELECT id_registro_HI as id_registro, 
        SCORE as score_lay, 
        FECHA_ASIGNACION as fecha_ejecucion, 
        FECHA_REVISION as fch_revision, 
        ESTATUS as estatus, 
        CLASIFICACION_DESVIOS as patron
      FROM `{lay}` 
      WHERE REGLAS = "NuevosM" and TRAMITE_UNICO='1'
      UNION ALL 
        SELECT id_registro_HI as id_registro, 
        SCORE as score_lay, 
        FECHA_ASIGNACION as fecha_ejecucion, 
        FECHA_REVISION as fch_revision, 
        ESTATUS as estatus, 
        CLASIFICACION_DESVIOS as patron
      FROM `{lay_20}`
      WHERE REGLAS = "NuevosM" and TRAMITE_UNICO='1'
      AND FECHA_ASIGNACION BETWEEN '{fecha_inicial}' AND '{fecha_final}'
    )
    SELECT *
    FROM ALERTAS
    LEFT JOIN CONSUMO USING(id_registro, fch_registro)
    LEFT JOIN LAY USING(id_registro, fecha_ejecucion)
    WHERE score_alerta is not null and num=1
    """

  else:
    raise Exception('El tipo de query debe ser total o alerta')


  return query

def query_test(alertas_train, evaluacion_train, lay_train, lay_train_test):
  query = f"""
  WITH ALERTAS AS (
    SELECT distinct id_registro,  fch_registro, model_name as modelo_hits
    FROM `{alertas_train}`),
  EVAL AS (
  SELECT id_registro, fch_registro, score_1, model_name as modelo
  FROM `{evaluacion_train}`
  ),
  LAY AS (
    SELECT distinct id_registro, fch_registro, estatus, CLASIFICACION_DE_DESVIOS as patron, train
    FROM `{lay_train_test}` 
    LEFT JOIN (SELECT id_registro, fch_registro, "Train" as train
    FROM `{lay_train}`)
    USING(id_registro, fch_registro)
    )
  SELECT LAY.id_registro, fch_registro, score_1, estatus, patron, modelo, modelo_hits, ifnull(train, "test") as train_test
  from LAY
  left join ALERTAS USING(id_registro, fch_registro)
  left join EVAL USING(id_registro, fch_registro)
  """
  return query


def query_alertas(fecha_inicial, fecha_final, lay, alertas, alertas_qa):
  query = f"""
  WITH QATAB AS (
  SELECT distinct id_registro, cast(fch_registro as date) as fch_registro_qa, cast(fecha_ejecucion as date) as fch_ejecucion, score_1 as score_qa, model_name as model_name_qa
  FROM `{alertas_qa}` 
  WHERE fch_registro between '2021-01-18' and '2021-02-04'),
  PROTAB AS (
  SELECT distinct id_registro, cast(fch_registro as date) as fch_registro_pro, cast(fecha_ejecucion as date) as fch_ejecucion, score_1 as score_pro, model_name as model_name_pro
  FROM `{alertas}` 
  ),
  LAY AS (
    SELECT id_registro_HI as id_registro, 
      SCORE as score_1, 
      FECHA_ASIGNACION as fch_asignacion, 
      FECHA_REVISION as fch_revision, 
      ESTATUS as estatus, 
      CLASIFICACION_DESVIOS as patron
    FROM `{lay}` 
    WHERE REGLAS = "NuevosM" and FECHA_ASIGNACION between '{fecha_inicial}' and '{fecha_final}' and TRAMITE_UNICO='1'
    ),
  ALERT AS (
  SELECT * , 
    CASE 
    when score_qa is null then "pro" 
    when score_pro is null then "qa" else "qa_pro" 
    end AS ambiente,
    concat(ifnull(model_name_qa, ""),"-", ifnull(model_name_pro, "")) as model_name,
    ifnull(fch_registro_pro, fch_registro_qa) as fch_registro
  FROM QATAB
  full JOIN PROTAB USING (id_registro, fch_ejecucion)
  )
  SELECT LAY.id_registro, fch_registro, fch_asignacion, fch_revision, score_qa, score_pro, score_1, estatus, patron, model_name_qa, model_name_pro, model_name, ambiente
  from LAY
  left join ALERT on LAY.id_registro = ALERT.id_registro AND fch_asignacion = fch_ejecucion
  """
  return query
