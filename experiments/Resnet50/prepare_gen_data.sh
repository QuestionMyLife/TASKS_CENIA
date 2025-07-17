#!/bin/bash

# Normal
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/normal" --task_idx $i --train_size 30_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --image_size 224
done

# Rigid- Triangles + Color
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/rigid_triangles_color" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode rigid --n_sides 3 --color True --image_size 224
done

# Rigid- Rectangles
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/rigid_rectangles" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode rigid --n_sides 4 --image_size 224
done

# Smooth low Fourier terms
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/smooth_low_fourier" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode smooth --fourier_terms 2 --image_size 224
done

# Symm + short
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/symm_short" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode symm --color True --max_size 0.3 --min_size 0.1 --image_size 224
done

# Normal con radio pequeño
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/normal_small_radius" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --radius 0.3 --image_size 224
done

# Rigid- Hexagons
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/rigid_hexagons" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode rigid --n_sides 6 --image_size 224
done

# Rigid- Decagons + Color
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/rigid_decagons_color" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode rigid --n_sides 10 --color True --image_size 224
done

# Symm + color + big
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/symm_color_big" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode symm --color True --max_size 0.6 --min_size 0.4 --image_size 224
done

# Smooth mid Fourier terms + color
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/smooth_mid_fourier_color" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode smooth --fourier_terms 4 --color True --image_size 224
done

# Smooth high Fourier terms
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/smooth_high_fourier" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode smooth --fourier_terms 32 --image_size 224
done

# Irregular pentagons + color
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/irregular_pentagons_color" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode rigid --rigid_type irregular --color True --n_sides 5 --image_size 224
done

# Irregular hexagons
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/irregular_hexagons" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode rigid --rigid_type irregular --n_sides 6 --image_size 224
done
# Irregular decagons + color
for i in {24..32}; do
  python generate_dataset_parallel.py --data_dir "experiments/images/irregular_decagons_color" --task_idx $i --train_size 15_000 --val_size 2_000 --test_size 10_000 --num_workers 30 --shape_mode rigid --rigid_type irregular --color True --n_sides 10 --image_size 224
done

