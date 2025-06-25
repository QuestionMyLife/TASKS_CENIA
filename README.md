
# A Benchmark for Efficient and Compositional Visual Reasoning

This reposity details the Compositional Visual Relations (CVR) benchmark. 
(Incluir funcionalidades antiguas?)
## Funcionalidades nueva clase `Shape`

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

