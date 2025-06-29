# TASKS_CENIA

Este repositorio contiene un benchmark para la generación y evaluación de tareas de razonamiento visual composicional, implementando en python el benchmark SVRT y extendiendolo con nuevas tareas y modos de generación de figuras.

Ciertas funciones fueron recuperadas y modificadas del siguiente repositorio:

<cite> Serre lab, CVR, (2022), GitHub repository, https://github.com/serre-lab/CVR </cite>

El cual esta relacionado a la siguiente publicación:

 <cite>@article{zerroug2022benchmark,
  title={A Benchmark for Compositional Visual Reasoning},
  author={Zerroug, Aimen and Vaishnav, Mohit and Colin, Julien and Musslick, Sebastian and Serre, Thomas},
  journal={arXiv preprint arXiv:2206.05379},
  year={2022}
}</cite>

## Dependencias

Para ejecutar este proyecto es necesario contar con las siguientes librerías de Python:

- **numpy**: Operaciones numéricas y manejo de arrays.
- **opencv-python** (`cv2`): Procesamiento y renderizado de imágenes.
- **Pillow** (`PIL`): Manipulación y guardado de imágenes.
- **matplotlib**: Utilizada para graficar y manipular contornos.
- **shapely**: Operaciones geométricas avanzadas (polígonos, inclusión, intersección, etc.).
- **scipy**: Utilizado en algunas tareas geométricas avanzadas.
- **argparse** y **logging**: Incluidas en la biblioteca estándar de Python.

### Instalación

Todas las dependencias necesarias pueden instalarse ejecutando el siguiente comando en la terminal:

```sh
pip install numpy opencv-python Pillow matplotlib shapely scipy
```
## Generacion de datos

Para generar los datos debe ejecutarse el archivo `generate_dataset_update`. Si no se entrega ningun argumento, el codigo se ejecuta con los valores por defecto. Los argumentos son especificados en el siguiente punto en la descripción detallada del archivo. Ejemplos concretos de comandos de generacion de datos pueden ser encontrados en `test_tasks.txt`. No obstante, a continuacion incluimos un comando para generar todos los datos para todas las tareas con los valores por defecto.
 
```sh
python generate_dataset_update.py --task_idx 0
```
## Descripción de Archivos Principales

### generate_dataset_update.py

Script principal para la generación de datasets de tareas visuales. Permite seleccionar la tarea, el modo de figura, los tamaños, colores y otros hiperparámetros mediante argumentos de línea de comandos. Utiliza las funciones de generación de tareas y figuras definidas en `data_generation/tasks_generation_cenia.py` y `data_generation/shape.py`.

- **Entradas:**
  - `--seed`: Semilla para la generación aleatoria (**int**, valor por defecto: `0`).
  - `--data_dir`: Ruta donde se guardarán los datos generados (**str**, valor por defecto: `'../cvrt_data/'`).
  - `--task_idx`: Índice de la tarea a generar (**int**, **requerido**).
  - `--train_size`: Número de ejemplos para el set de entrenamiento (**int**, valor por defecto: `4`).
  - `--val_size`: Número de ejemplos para el set de validación (**int**, valor por defecto: `4`).
  - `--test_size`: Número de ejemplos para el set de test (**int**, valor por defecto: `4`).
  - `--image_size`: Tamaño (alto y ancho) de las imágenes generadas en píxeles (**int**, valor por defecto: `128`).
  - Argumentos para generación de figuras:
    - `--shape_mode`: Modo de generación de la figura (`normal`, `rigid`, `smooth`, `symm`) (**str**, valor por defecto: `'normal'`).
    - `--radius`: Radio base de la figura (**float**, valor por defecto: `0.5`).
    - `--hole_radius`: Radio del agujero interior de la figura (**float**, valor por defecto: `0.05`).
    - `--n_sides`: Número de lados para polígonos (**int**, valor por defecto: `5`).
    - `--fourier_terms`: Número de términos de Fourier para suavizado (**int**, valor por defecto: `4`).
    - `--max_size`: Tamaño máximo de la figura (**float**, valor por defecto: `0.4`).
    - `--min_size`: Tamaño mínimo de la figura (**float**, valor por defecto: `0.2`).
    - `--color`: Si las figuras serán a color (**bool**, valor por defecto: `False`).
    - `--rigid_type`: Tipo de figura rígida (`polygon`, etc.) (**str**, valor por defecto: `'polygon'`).
    - `--symm_rotation`: Si se permite rotación simétrica en figuras de simetría (**bool**, valor por defecto: `False`).
- **Salida:** imágenes generadas en carpetas organizadas por tarea y split (train/val/test).

### test_tasks.txt

Archivo de texto con ejemplos de comandos para probar diferentes tareas y modos de generación de figuras. Útil como referencia rápida para ejecutar pruebas.

### data_generation/shape.py

La clase `Shape` es la base para la generación, manipulación y transformación de figuras geométricas utilizadas en las tareas del benchmark. A continuación se listan y describen sus métodos:

- **`__init__(...)`**: Constructor de la clase. Inicializa los parámetros de la figura (radio, hueco, simetría, etc.) y genera una figura aleatoria si `randomize=True`.
- **`generate_part(...)`**: Genera los puntos base de la figura, asegurando que estén dentro del radio y fuera del hueco central.
- **`generate_part_part(...)`**: Genera una sección de la figura entre dos puntos, subdividiendo si es necesario para suavizar el contorno.
- **`randomize()`**: Genera una nueva figura aleatoria, normalizándola y centrando sus puntos.
- **`smooth(n_sampled_points=100, fourier_terms=20)`**: Suaviza el contorno de la figura utilizando una aproximación de Fourier.
- **`rigid_transform(type='polygon', points=3, rotate=0)`**: Transforma la figura en un polígono regular, irregular o flecha, con opción de rotación aleatoria.
- **`symmetrize(rotate=0)`**: Convierte la figura en una figura simétrica respecto al eje vertical, con opción de rotación.
- **`flip_diag()`**: Intercambia los ejes x e y de la figura (simetría diagonal).
- **`get_contour()`**: Devuelve el contorno de la figura como un array de puntos (cerrando el contorno).
- **`rotate(alpha)`**: Rota la figura un ángulo `alpha` (en radianes).
- **`get_bb()`**: Devuelve el ancho y alto del bounding box de la figura.
- **`scale(s)`**: Escala la figura por un factor `s` (puede ser escalar o tupla para cada eje).
- **`set_wh(wh)`**: Ajusta el ancho y alto de la figura a los valores dados.
- **`set_size(s)`**: Ajusta el tamaño global de la figura.
- **`clone()`**: Devuelve una copia profunda de la figura actual.
- **`flip()`**: Refleja la figura respecto al eje vertical.
- **`subsample()`**: Añade puntos intermedios al contorno para suavizarlo y lo perturba aleatoriamente.
- **`reset()`**: Revierte todas las transformaciones aplicadas a la figura, restaurando su estado original.
- **`get_hole_radius()`**: Calcula el radio máximo del círculo que cabe dentro de la figura (usando transformada de distancia).

#### Clase derivada: `ShapeCurl`
- **`ShapeCurl.randomize()`**: Variante de `randomize()` que aplica un suavizado adicional al contorno de la figura.

Estas funcionalidades permiten crear figuras complejas, transformarlas, suavizarlas, clonarlas y adaptarlas a diferentes tareas de razonamiento visual.

### data_generation/tasks_cenia_desc.py
Contiene descripciones de las tareas SVRT y extendidas, incluyendo el número de objetos, detalles de las categorías y referencias a las funciones de generación de cada tarea.

### data_generation/tasks_generation_cenia.py
Archivo que implementa todas las funciones de generación de tareas visuales (SVRT y extendidas). Cada función genera ejemplos positivos y negativos para una tarea específica, utilizando las utilidades de generación y decoración de figuras.

- Define la función create_shape para crear figuras según el modo (normal, rigid, smooth, symm).
- Implementa funciones como task_svrt_1, task_svrt_2, ..., task_MTS, task_SD, etc.
- Registra todas las tareas en la lista TASKS_SVRT para su uso desde el script principal.
### data_generation/utils.py
Funciones auxiliares para la generación y manipulación de escenas y figuras. Incluye utilidades para muestreo de posiciones, colores, chequeo de condiciones geométricas y renderizado de imágenes.

- **Funciones principales:**
  - **`hsv_to_rgb(h, s, v)`**: Convierte un color en espacio HSV a RGB.
  - **`sample_position_inside_1(s1, s2, scale, n_candidates=200)`**: Devuelve centros posibles para ubicar una figura s2 (escalada) completamente dentro de s1.
  - **`sample_position_outside_1(s1, s2, scale, n_candidates=200)`**: Devuelve centros posibles para ubicar una figura s2 (escalada) completamente fuera de s1.
  - **`sample_position_inside_many(s1, shapes, scales, n_candidates=500)`**: Devuelve configuraciones de centros para varias figuras dentro de s1, sin solapamiento.
  - **`sample_positions_square(size)`**: Genera posiciones para 4 objetos formando un cuadrado.
  - **`squared_distance(p1, p2)`**: Calcula la distancia euclidiana al cuadrado entre dos puntos.
  - **`check_square(xy)`**: Verifica si 4 puntos forman un cuadrado.
  - **`sample_positions_equidist(size, max_attempts=100, max_inner_attempts=50)`**: Genera posiciones para 4 objetos donde dos pares tienen igual distancia.
  - **`sample_positions_bb(size, n_sample_min=1, max_tries=10, n_samples_over=100)`**: Genera posiciones aleatorias para objetos de cierto tamaño sin solapamiento.
  - **`sample_positions_align(size)`**: Genera posiciones alineadas para varios objetos.
  - **`sample_positions_symmetric_pairs(size, ...)`**: Genera posiciones para pares de objetos simétricos respecto al eje vertical.
  - **`sample_points_in_circle(n, radius=0.2, center=(0.5,0.5), on_edge=False)`**: Genera n puntos dentro o sobre el borde de un círculo.
  - **`sample_positions_circle(sizes, ...)`**: Genera posiciones para objetos distribuidos en círculo, útil para tareas de "odd one out".
  - **`compute_inscribed_circle(shape, resolution=100)`**: Calcula el mayor círculo inscrito en una figura.
  - **`sample_random_colors(n_samples)`**: Genera n colores aleatorios en espacio HSV.
  - **`sample_contact_many(shapes, sizes, image_dim=128, a=None)`**: Genera posiciones para que varias figuras estén en contacto.
  - **`render_cv(xy, size, shapes, color=None, image_size=128)`**: Renderiza una escena con OpenCV a partir de las posiciones, tamaños y colores.
  - **`render_scene_safe(xy, size, shape, color, image_size=128)`**: Renderiza múltiples figuras en una sola imagen, asegurando el formato correcto de entrada.

Estas funciones auxiliares son utilizadas en diversas tareas.


