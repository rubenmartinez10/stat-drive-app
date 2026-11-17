# STAT-DRIVE: ¿cuánto tardo en llegar?

Aplicación web que estima el tiempo de un viaje (por ejemplo, de la casa a la UCA) a partir de la hora de salida, la distancia, el tipo de transporte y el nivel de tráfico. El modelo se entrena con viajes reales medidos por estudiantes y se reentrena cada noche con los datos nuevos que llegan por un formulario.

Proyecto final de **Probabilidad y Estadística** · Universidad Centroamericana José Simeón Cañas (UCA), Departamento de Matemática · Sección 01 · Ciclo 02-2026.

**Aplicación:** https://rubenmartinez10.github.io/stat-drive-app/

## Cómo funciona

```
Google Form  →  Hoja de respuestas (CSV publicado)  →  GitHub Actions (cada noche)
                                                          │  train.py reentrena
                                                          ▼
                                   model.json  →  index.html (GitHub Pages) calcula en el navegador
```

1. Cualquier persona registra un viaje real en el formulario.
2. Cada día a las **3:00 a. m. (hora de El Salvador)**, una tarea de GitHub Actions lee la hoja, descarta datos inválidos, une los viajes nuevos con los de `viajes_base.csv` y reentrena.
3. El resultado se guarda en `model.json`. La página lee ese archivo y calcula la predicción en el navegador, sin servidor.

## El modelo

Regresión lineal múltiple, ajustada por mínimos cuadrados, que predice `real_min` (minutos reales del viaje).

| Variable | Descripción |
|---|---|
| `distancia_km` | Distancia del viaje |
| `hora_salida` | Hora de salida en formato decimal (14:30 = 14.5) |
| `nivel_trafico` | 0 = leve, 1 = moderado, 2 = mucho (según una app de mapas al salir) |
| `es_regreso` | 1 si el viaje es UCA → casa |
| `es_otra` | 1 si es un viaje a otra actividad |
| `es_bus` | 1 si el transporte es bus (0 si es carro) |

**Evaluación.** Validación cruzada de 10 pliegues. Con los 137 viajes iniciales, el error absoluto medio fue de aproximadamente **15.7 min**. El valor vigente se muestra en la página y se guarda en `model.json` (`mae_cv`).

**Rango de la predicción.** Se calcula con percentiles empíricos de los errores absolutos de la validación cruzada: `e50` (la mitad de los casos queda dentro de ese margen) y `e90` (el 90 %). La página muestra ambos: "lo más probable" y "casi seguro".

## Datos

- **Base inicial:** 137 viajes de 4 estudiantes, cronometrados entre el 16/10/2026 y el 09/11/2026 (`viajes_base.csv`, sin nombres ni fechas).
- **Nivel de tráfico en la base:** el registro original contaba incidencias por viaje. Para unificarlo con el formulario se convirtió así: 0 incidencias = leve, 1 = moderado, 2 o más = mucho. Es una aproximación. Los datos del formulario sí traen el nivel de tráfico real.
- **Datos nuevos:** llegan por el formulario. Solo se aceptan transporte `carro` o `bus`, distancia entre 0.5 y 150 km, tiempo real entre 1 y 400 min y hora entre 0 y 24.

## Estructura del repositorio

| Archivo | Función |
|---|---|
| `index.html` | Interfaz de la aplicación (HTML, CSS y JavaScript en un solo archivo) |
| `train.py` | Carga y valida los datos, entrena el modelo y escribe `model.json` |
| `model.json` | Coeficientes del modelo y métricas (lo genera el entrenamiento) |
| `viajes_base.csv` | Datos iniciales |
| `.github/workflows/reentrenar.yml` | Tarea programada que reentrena cada noche |

## Puesta en marcha

### 1. Formulario de Google

Crear un formulario con estas preguntas. Los **títulos deben ser exactamente estos**:

| Título | Tipo | Opciones o rango |
|---|---|---|
| `tipo` | Desplegable | `ida`, `regreso`, `otra` |
| `hora_salida` | Hora | |
| `distancia_km` | Número | 0.5 a 150 |
| `nivel_trafico` | Desplegable | `Leve`, `Moderado`, `Mucho` |
| `transporte` | Desplegable | `Carro`, `Bus` |
| `real_min` | Número | 1 a 400 |

No se recogen nombres ni ubicaciones exactas.

### 2. Publicar la hoja de respuestas

En la hoja vinculada al formulario: *Archivo → Compartir → Publicar en la web*, elegir la pestaña de respuestas, formato **CSV**, y copiar el enlace.

### 3. Configurar GitHub

1. En *Settings → Secrets and variables → Actions*, crear el secreto `FORM_CSV_URL` con ese enlace.
2. En *Settings → Actions → General → Workflow permissions*, activar **Read and write permissions**.
3. En *Settings → Pages*, publicar desde la rama `main`, carpeta raíz.
4. En `index.html`, el enlace al formulario está en el botón "Registrar mi viaje real".
5. Para entrenar de inmediato: pestaña *Actions → reentrenar → Run workflow*.

### Ejecución local

```powershell
pip install pandas numpy
python train.py                                   # solo con viajes_base.csv
$env:FORM_CSV_URL = "https://..."                  # opcional: incluir el formulario
python train.py
```

## Limitaciones

- **Pocos datos y pocas personas.** El modelo se construyó con 4 estudiantes. Al dejar a cada persona fuera de la validación, el error medio estuvo entre 12 y 29 min, por lo que para alguien nuevo el error puede ser mayor que el reportado.
- **El efecto del bus** se estima con solo dos personas, por lo que puede reflejar sus rutas y no el transporte en general.
- **Rango de distancias.** Las predicciones fuera del rango observado (2 a 120 km en la base) son poco confiables. La aplicación lo advierte.
- **Solo carro y bus.** No hay datos de moto ni de caminata.
- **Nivel de tráfico** de la base es una conversión desde incidencias (ver Datos).
- **Medición manual.** Los tiempos se cronometraron a mano, en minutos enteros.
- **Calidad del formulario.** Un dato mal capturado pero dentro de los rangos permitidos entra al entrenamiento.

Esta herramienta es un ejercicio académico y no sustituye a una aplicación de navegación.

## Privacidad

Los datos son anónimos: la base inicial no contiene nombres ni fechas, y el formulario no pide datos personales. La hoja publicada es visible para quien tenga el enlace, por lo que no debe contener información personal.

## Uso de inteligencia artificial

El desarrollo de la aplicación y del código de entrenamiento se realizó por el equipo, utilizando Claude (Anthropic) como herramienta de apoyo durante el proceso. El código generado o sugerido fue revisado, adaptado y probado por el equipo. La recolección de datos, la selección de variables, las decisiones de diseño y la interpretación de los resultados fueron realizadas por el equipo.

## Autoría y contribuciones

Proyecto desarrollado por el equipo. La implementación de la aplicación y del código de entrenamiento estuvo a cargo de Rubén Eliseo Martínez Pérez, con participación de los demás integrantes en las diferentes etapas del proyecto.

## Licencia

Este proyecto se distribuye bajo la Licencia MIT. Consulte el archivo [`LICENSE`](LICENSE) para conocer los términos completos de uso, modificación y distribución del código.
