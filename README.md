
# Segmentación de imágenes mediante Watershed

Herramienta desarrollada en Python para segmentar imágenes mediante el algoritmo Watershed controlado por marcadores. Permite separar objetos conectados, configurar los parámetros de segmentación y comparar los resultados mediante métricas cuantitativas y representaciones gráficas.

## Instalación

Se requiere Python 3 y las siguientes bibliotecas:

- OpenCV
- NumPy
- Matplotlib

Para instalar las dependencias, ejecutar:

```bash
python -m pip install -r requirements.txt
```

## Modo de uso

### Procesar una imagen

```bash
python watershed_eval.py imagen.jpg
```

### Procesar varias imágenes

```bash
python watershed_eval.py imagen1.jpg imagen2.jpg
```

### Procesar una carpeta completa

```bash
python watershed_eval.py imagenes/
```

### Configurar el método de umbralización

El programa permite utilizar los métodos de Otsu y Kapur.

```bash
python watershed_eval.py imagen.jpg --method otsu
```

```bash
python watershed_eval.py imagen.jpg --method kapur
```

### Configurar los umbrales de distancia

Es posible evaluar diferentes valores de umbral para la generación de marcadores y comparar sus resultados.

```bash
python watershed_eval.py imagen.jpg --thresholds 0.3 0.5 0.7
```

### Mostrar los resultados en pantalla

```bash
python watershed_eval.py imagen.jpg --show
```

### Cambiar la carpeta de salida

```bash
python watershed_eval.py imagen.jpg -o mis_resultados
```

## Parámetros de ejecución

| Parámetro | Descripción |
|---|---|
| `inputs` | Imágenes o carpetas que se desean procesar. |
| `--method` | Método de umbralización: `otsu` o `kapur`. |
| `--thresholds` | Valores de umbral para la transformada de distancia. |
| `--landscape` | Superficie utilizada por Watershed: `image` o `distance`. |
| `--show` | Muestra las figuras en pantalla. |
| `--output`, `-o` | Define la carpeta de salida. |
| `--version` | Identificador de la configuración utilizada. |
| `--help` | Muestra todos los parámetros disponibles. |

## Funcionamiento

El programa realiza un preprocesamiento de la imagen mediante conversión a escala de grises, suavizado, umbralización y operaciones morfológicas.

Posteriormente, calcula la transformada de distancia y genera los marcadores utilizados por Watershed para identificar y separar las regiones de interés.

La segmentación se ejecuta con diferentes valores de umbral, permitiendo comparar el comportamiento del algoritmo y detectar posibles pérdidas de regiones o sobresegmentación.

## Archivos generados

Los resultados se almacenan de manera predeterminada en la carpeta `resultados/`.

| Archivo | Contenido |
|---|---|
| `*_fig_pipeline.png` | Representación gráfica de las etapas del procesamiento. |
| `*_fig_watershed.png` | Comparación visual de las segmentaciones obtenidas. |
| `*_histograma_info.json` | Información asociada a la umbralización de la imagen. |
| `metricas_*.csv` | Métricas de evaluación para cada configuración. |
| `parametros_*.json` | Parámetros utilizados durante la ejecución. |
| `resumen_*.md` | Resumen de los resultados obtenidos. |

## Interpretación de resultados

El programa compara las componentes iniciales con las regiones obtenidas después de aplicar Watershed y clasifica los resultados mediante los siguientes criterios:

- **Mejora candidata:** se identifica una posible mejora en la separación de objetos.
- **Sin mejora relevante:** no se consigue una separación significativa.
- **Contraproducente:** se detecta pérdida de regiones o posible sobresegmentación.
- **Revisar visualmente:** el resultado requiere una inspección adicional.

Estas clasificaciones permiten comparar configuraciones y seleccionar los parámetros más adecuados para cada imagen.

## Ejemplo de ejecución

```bash
python watershed_eval.py imagenes/ --method kapur --thresholds 0.3 0.5 0.7 --show -o resultados
```

Este comando procesa todas las imágenes compatibles de la carpeta indicada, utiliza el método de Kapur, evalúa tres umbrales de distancia, muestra las figuras y almacena los resultados en la carpeta de salida.
