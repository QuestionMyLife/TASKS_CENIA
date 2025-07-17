import argparse
import logging
import os
from multiprocessing import Pool
from time import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from data_generation.tasks_generation_cenia import TASKS_SVRT
from data_generation.utils import render_scene_safe

TASKS_IDX = {
    1: "task_svrt_1", 2: "task_svrt_2", 3: "task_svrt_3", 4: "task_svrt_4",
    5: "task_svrt_5", 6: "task_svrt_6", 7: "task_svrt_7", 8: "task_svrt_8",
    9: "task_svrt_9", 10: "task_svrt_10", 11: "task_svrt_11", 12: "task_svrt_12",
    13: "task_svrt_13", 14: "task_svrt_14", 15: "task_svrt_15", 16: "task_svrt_16",
    17: "task_svrt_17", 18: "task_svrt_18", 19: "task_svrt_19", 20: "task_svrt_20",
    21: "task_svrt_21", 22: "task_svrt_22", 23: "task_svrt_23", 24: "task_MTS",
    25: "task_SD", 26: "task_SOSD", 27: "task_RMTS", 28: "task_sym_classification",
    29: "task_sym_MTS", 30: "task_sym_SD", 31: "task_sym_SOSD",
    32: "task_sym_RMTS"
}

def generate_single_example(args):
    i, split, task_name, image_size, base_kwargs, data_path, example_seed = args

    np.random.seed(example_seed)  # 🌱 Semilla única por imagen

    task_fn_base = next(fn for name, fn in TASKS_SVRT if name == task_name)
    sample_neg, sample_pos = task_fn_base(**base_kwargs)

    if isinstance(sample_pos, bool):
        return  # Tarea no implementada

    task_path = os.path.join(data_path, task_name)
    for label, (xy, size, shape, color) in enumerate([sample_neg, sample_pos]):
        image = render_scene_safe(xy, size, shape, color, image_size=image_size)
        img = Image.fromarray(image).convert('RGB')
        save_path = os.path.join(task_path, split, f'class_{label}_{i:07d}.png')
        img.save(save_path)

def generate_dataset_parallel(task_name, base_kwargs, data_path, image_size, seed,
                              train_size, val_size, test_size, num_workers=2):
    task_path = os.path.join(data_path, task_name)
    os.makedirs(os.path.join(task_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(task_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(task_path, 'test'), exist_ok=True)

    splits = {
        'train': (seed, train_size),
        'val': (seed + train_size + 1_000_000, val_size),
        'test': (seed + train_size + val_size + 2_000_000, test_size),
    }


    for split, (split_seed, n_samples) in splits.items():
        print(f"[{task_name}] Generando {n_samples} ejemplos para {split}...")

        args_list = [
            (i, split, task_name, image_size, base_kwargs, data_path, split_seed + i)
            for i in range(n_samples)
        ]

        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(generate_single_example, args_list),
                      total=n_samples, desc=f"{task_name} | {split}"))

    print(f"[{task_name}] ✔ Dataset generado en: {task_path}")
    
    
def build_kwargs(args):
    return dict(
        shape_mode=args.shape_mode,
        radius=args.radius,
        hole_radius=args.hole_radius,
        n_sides=args.n_sides,
        fourier_terms=args.fourier_terms,
        max_size=args.max_size,
        min_size=args.min_size,
        color=args.color,
        rigid_type=args.rigid_type,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='../cvrt_data/')
    parser.add_argument('--task_idx', type=int, required=True)
    parser.add_argument('--train_size', type=int, default=4)
    parser.add_argument('--val_size', type=int, default=4)
    parser.add_argument('--test_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--shape_mode', type=str, default='normal')
    parser.add_argument('--radius', type=float, default=0.5)
    parser.add_argument('--hole_radius', type=float, default=0.05)
    parser.add_argument('--n_sides', type=int, default=5)
    parser.add_argument('--fourier_terms', type=int, default=4)
    parser.add_argument('--max_size', type=float, default=0.4)
    parser.add_argument('--min_size', type=float, default=0.2)
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--rigid_type', type=str, default='polygon')
    parser.add_argument('--symm_rotation', type=bool, default=False)

    args = parser.parse_args()
    logging.info(f'JOB PID {os.getpid()}')
    t0 = time()

    base_kwargs = build_kwargs(args)
    base_sym_kwargs = base_kwargs.copy()
    base_sym_kwargs['symm_rotation'] = args.symm_rotation

    if args.task_idx == 0:
        for task_idx, task_name in TASKS_IDX.items():
            # Incluir argumento de rotación simétrica para tareas de simetría
            if 'sym' in task_name:
                try:
                    generate_dataset_parallel(
                        task_name=task_name,
                        base_kwargs=base_sym_kwargs,
                        data_path=args.data_dir,
                        image_size=args.image_size,
                        seed=args.seed,
                        train_size=args.train_size,
                        val_size=args.val_size,
                        test_size=args.test_size,
                        num_workers=args.num_workers
                    )
                except Exception as e:
                    print(f"❌ Error en {task_name}: {e}")
            else:
                try:
                    generate_dataset_parallel(
                        task_name=task_name,
                        base_kwargs=base_kwargs,
                        data_path=args.data_dir,
                        image_size=args.image_size,
                        seed=args.seed,
                        train_size=args.train_size,
                        val_size=args.val_size,
                        test_size=args.test_size,
                        num_workers=args.num_workers
                    )
                except Exception as e:
                    print(f"❌ Error en {task_name}: {e}")
    else:
        try:
            task_name = TASKS_IDX[args.task_idx]
        except KeyError:
            raise ValueError(f"Tarea con índice {args.task_idx} no registrada.")
        if 'sym' in task_name:
            generate_dataset_parallel(
                task_name=task_name,
                base_kwargs=base_sym_kwargs,
                data_path=args.data_dir,
                image_size=args.image_size,
                seed=args.seed,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                num_workers=args.num_workers
            )
        else:
            generate_dataset_parallel(
                task_name=task_name,
                base_kwargs=base_kwargs,
                data_path=args.data_dir,
                image_size=args.image_size,
                seed=args.seed,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                num_workers=args.num_workers
            )

    print(f"✅ Finalizado en {time() - t0:.2f} segundos.")
    print(f"✅ Finalizado en {time() - t0:.2f} segundos.")
