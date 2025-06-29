# TASKS_CENIA

Este repositorio contiene un benchmark para la generación y evaluación de tareas de razonamiento visual composicional, implementando en python el benchmark SVRT y extendiendolo con nuevas tareas y modos de generación de figuras.

## Descripción de Archivos Principales

### generate_dataset_update.py

Script principal para la generación de datasets de tareas visuales. Permite seleccionar la tarea, el modo de figura, los tamaños, colores y otros hiperparámetros mediante argumentos de línea de comandos. Utiliza las funciones de generación de tareas y figuras definidas en `data_generation/tasks_generation_cenia.py` y `data_generation/shape.py`.

- **Entradas:** 
  - `--seed`: Semilla para la generación aleatoria (int, default: 0).
  - `--data_dir`: Ruta donde se guardarán los datos generados (str, default: '../cvrt_data/').
  - `--task_idx`: Índice de la tarea a generar (int, requerido).
  - `--train_size`: Número de ejemplos para el set de entrenamiento (int, default: 4).
  - `--val_size`: Número de ejemplos para el set de validación (int, default: 4).
  - `--test_size`: Número de ejemplos para el set de test (int, default: 4).
  - `--image_size`: Tamaño (alto y ancho) de las imágenes generadas en píxeles (int, default: 128).
  - Argumentos para generación de figuras:
    - `--shape_mode`: Modo de generación de la figura (`normal`, `rigid`, `smooth`, `symm`) (str, default: 'normal').
    - `--radius`: Radio base de la figura (float, default: 0.5).
    - `--hole_radius`: Radio del agujero interior de la figura (float, default: 0.05).
    - `--n_sides`: Número de lados para polígonos (int, default: 5).
    - `--fourier_terms`: Número de términos de Fourier para suavizado (int, default: 4).
    - `--max_size`: Tamaño máximo de la figura (float, default: 0.4).
    - `--min_size`: Tamaño mínimo de la figura (float, default: 0.2).
    - `--color`: Si las figuras serán a color (`True` o `False`) (bool, default: False).
    - `--rigid_type`: Tipo de figura rígida (`polygon`, etc.) (str, default: 'polygon').
    - `--symm_rotation`: Si se permite rotación simétrica en figuras de simetría (bool, default: False).
- **Salida:** imágenes generadas en carpetas organizadas por tarea y split (train/val/test).
- **Uso típico:**  
  ```sh
  python generate_dataset_update.py --task_idx 1 --shape_mode smooth --fourier_terms 10 --data_dir "dir"

### test_tasks.txt
Archivo de texto con ejemplos de comandos para probar diferentes tareas y modos de generación de figuras. Útil como referencia rápida para ejecutar pruebas.

### data_generation/shape.py
Define la clase principal Shape y sus métodos para la generación, transformación y manipulación de figuras geométricas. Incluye funcionalidades como:

- Generación aleatoria de contornos.
- Suavizado mediante Fourier (smooth).
Transformaciones rígidas (polígonos regulares, irregulares, flechas).
- Simetrización y rotación.
- Métodos auxiliares para escalar, clonar, reflejar y obtener bounding boxes.
### data_generation/tasks_cenia_desc.py
Contiene descripciones de las tareas SVRT y extendidas, incluyendo el número de objetos, detalles de las categorías y referencias a las funciones de generación de cada tarea.

### data_generation/tasks_generation_cenia.py
Archivo central que implementa todas las funciones de generación de tareas visuales (SVRT y extendidas). Cada función genera ejemplos positivos y negativos para una tarea específica, utilizando las utilidades de generación y decoración de figuras.

- Define la función create_shape para crear figuras según el modo (normal, rigid, smooth, symm).
- Implementa funciones como task_svrt_1, task_svrt_2, ..., task_MTS, task_SD, etc.
- Registra todas las tareas en la lista TASKS_SVRT para su uso desde el script principal.
### data_generation/utils.py
Funciones auxiliares para la generación y manipulación de escenas y figuras:

- Muestreo de posiciones (alineadas, en círculo, cuadrado, sin solapamiento, etc.).
- Muestreo de colores aleatorios.
- Renderizado seguro de escenas a imágenes.
- Cálculo de círculos inscritos, chequeo de cuadrado, etc.
### Carpeta dir/
Contiene las carpetas de salida para cada tarea generada, organizadas por nombre de tarea (por ejemplo, task_svrt_1/, task_MTS/, etc.), y dentro de cada una, los splits train/, val/, test/ con las imágenes generadas.

# A Benchmark for Efficient and Compositional Visual Reasoning

## Funcionalidades clase `Shape`

* `smooth`: Suaviza la figura utilizando Fourier. Toma como argumentos 
  - `n_sampled_points`: número de puntos muestreados. Valor por defecto: 100.
  - `fourier_terms`: número de términos de Fourier. Valor por defecto: 20.

* `rigid_transform`: Transforma la figura en una figura rígida. Toma como argumentos
  - `type`: tipo de transformación. Por ahora, puede ser `'polygon'` para generar polígonos regulares, `'irregular'` para generar polígonos irregulares. Valor por defecto: `'polygon'`. 
  - `points`: número de vértices. Valor por defecto: 3.
  - `rotate`: flag que indica si se debe rotar la figura. En caso de ser `True`, la figura se rotará aleatoriamente. Valor por defecto: 0. 

* `symmetrize`: Genera una figura con simetría axial con respecto al eje $y$ a partir de la figura original. Toma como argumentos
  - `rotate`: flag que indica si se debe rotar la figura. En caso de ser `True`, la figura se rotará aleatoriamente. Valor por defecto: 0.


## Citation

```stex
@article{zerroug2022benchmark,
  title={A Benchmark for Compositional Visual Reasoning},
  author={Zerroug, Aimen and Vaishnav, Mohit and Colin, Julien and Musslick, Sebastian and Serre, Thomas},
  journal={arXiv preprint arXiv:2206.05379},
  year={2022}
}
```

