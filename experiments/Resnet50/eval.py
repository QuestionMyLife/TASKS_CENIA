# -*- coding: utf-8 -*-
"""
Extended evaluation script that, besides evaluating every (model, dataset)
combination, now adds **progress bars** so you can watch the work advance in real
‑time:

1. **Global progress bar** over the cartesian product *models × datasets*.
2. **Inner progress bar** over batches while each evaluation is running.
3. Still saves one bar‑plot (PNG) and one CSV **per model**, plus the global
   ``accuracy_summary.csv``.

Outputs are stored in the folder supplied via ``--out_dir`` (default: ``outputs``).

Author: ChatGPT (o3) – June 2025
"""

import os
from pathlib import Path
import argparse
from itertools import product

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # 🟢 progreso elegante en notebooks y terminales

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def construct_paths(model_name: str, dataset_name: str, base_dir: str = "experiments/Resnet50"):
    """Devuelve (model_path, data_path) según la convención de carpetas."""
    if model_name.endswith("_gen"):
        task_name = model_name.replace("_gen", "")
        model_dir = f"resnet50-{task_name}"
        model_file = f"resnet50_{model_name}.pth"
    else:
        model_dir = f"resnet50-{model_name}"
        model_file = f"resnet50_{model_name}.pth"
    model_path = os.path.join(base_dir, "models", model_dir, model_file)

    if dataset_name.startswith("task_"):
        data_path = os.path.join(base_dir, "images_processed", dataset_name)
    else:
        data_path = os.path.join(base_dir, "images_processed", dataset_name, model_name)
    return model_path, data_path


# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_name: str, dataset_name: str, batch_size: int = 128):
    """Evalúa *un* modelo sobre *un* dataset y devuelve un dict con resultados."""
    model_path, data_path = construct_paths(model_name, dataset_name)

    if not os.path.exists(model_path):
        return {"error": f"Modelo no encontrado: {model_path}", "model": model_name, "dataset": dataset_name, "accuracy": None}
    if not os.path.exists(data_path):
        return {"error": f"Dataset no encontrado: {data_path}", "model": model_name, "dataset": dataset_name, "accuracy": None}

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dir = os.path.join(data_path, "test")
    if not os.path.exists(test_dir):
        return {"error": f"Directorio de test no encontrado: {test_dir}", "model": model_name, "dataset": dataset_name, "accuracy": None}

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(test_dataset.classes)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    # Inner progress bar per batch (leave=False so only the outer bar remains after finish)
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"{model_name} → {dataset_name}", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total else 0.0
    return {"model": model_name, "dataset": dataset_name, "accuracy": accuracy, "correct": correct, "total": total, "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# MULTI‑EVALUATION WRAPPER WITH GLOBAL PROGRESS BAR ──────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_multiple(models, datasets, batch_size: int = 128):
    combos = list(product(models, datasets))
    results = []
    for model_name, dataset_name in tqdm(combos, desc="Model × Dataset", unit="comb"):
        results.append(evaluate_model(model_name, dataset_name, batch_size=batch_size))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT UTILITIES (CSV & PLOTS) ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def collect_successful(results):
    return [r for r in results if r["error"] is None]


def save_summary_csv(df: pd.DataFrame, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=True)
    print(f"📑 CSV general guardado en: {csv_path}")


def save_csv_per_model(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in df.columns:
        model_df = df[[model]].copy().rename(columns={model: "accuracy"})
        csv_path = out_dir / f"accuracy_{model}.csv"
        model_df.to_csv(csv_path, index_label="dataset")
        print(f"📑 CSV por modelo guardado en: {csv_path}")


def plot_per_model(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in df.columns:
        ax = df[model].plot(kind="bar", figsize=(10, 6))
        ax.set_title(f"Accuracy por dataset – {model}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = out_dir / f"accuracy_{model}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Gráfico guardado en: {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="experiments/Resnet50/outputs", help="Directorio para CSV y PNGs")
    args = parser.parse_args()

    # Lista de modelos y datasets (edítalas a tu gusto)
    models_to_evaluate = [
        "task_SD_gen",
        "task_MTS_gen",
        "task_RMTS_gen",
        "task_SOSD_gen",
        "task_sym_classification_gen",
        "task_sym_MTS_gen",
        "task_sym_SD_gen",
    ]

    datasets_to_evaluate = [
        "normal", "normal_small_radius", "smooth_low_fourier", "rigid_triangles_color", "symm_color_big",
        "rigid_decagons_color", "rigid_hexagons", "rigid_rectangles", "smooth_high_fourier", "smooth_mid_fourier_color", "symm_short",
        "irregular_decagons_color", "irregular_hexagons","irregular_pentagons_color"
    ]

    results = evaluate_multiple(models_to_evaluate, datasets_to_evaluate, batch_size=args.batch_size)
    successful = collect_successful(results)
    if not successful:
        print("❌ No hay resultados exitosos para graficar o guardar.")
        return

    # DataFrame: rows = dataset, columns = model, values = accuracy (0‑100).
    df = pd.DataFrame(successful).pivot(index="dataset", columns="model", values="accuracy") * 100.0

    out_dir = Path(args.out_dir)
    save_summary_csv(df, out_dir / "accuracy_summary.csv")
    save_csv_per_model(df, out_dir)
    plot_per_model(df, out_dir)


if __name__ == "__main__":
    main()
