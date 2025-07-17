# -*- coding: utf-8 -*-
"""
Evaluation utility **dedicated to CLIP‑ViT classifiers**.

Features
--------
1. **Global progress bar** for the full *(model, dataset)* grid.
2. **Inner progress bar** for batches while evaluating each model.
3. Saves one **bar plot** (`accuracy_<model>.png`) and one **CSV**
   (`accuracy_<model>.csv`) *per model*.
4. Saves the full accuracy matrix as ``accuracy_summary.csv``.

Expected folder structure
-------------------------
```
experiments/ClipViT/
├── models/
│   ├── task_SD_gen.pth
│   ├── task_MTS_gen.pth
│   └── …
└── images_processed/
    ├── normal/
    │   └── test/
    └── …
```

Author: ChatGPT (o3) – June 2025
"""

import os
from pathlib import Path
import argparse
from itertools import product

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPER ══════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

class ClipVisionClassifier(nn.Module):
    """Wrap CLIP's vision tower with a linear classification head."""

    def __init__(self, vision_model: CLIPModel, num_classes: int):
        super().__init__()
        self.vision_model = vision_model
        projection_dim = vision_model.config.hidden_size  # 768 for ViT‑B/32
        self.class_head = nn.Linear(projection_dim, num_classes)

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values)
        pooled = outputs.pooler_output  # (batch, hidden_size)
        return self.class_head(pooled)

# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS ══════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys if present."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# ════════════════════════════════════════════════════════════════════════════
# PATHS ══════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

def construct_paths(model_name: str, dataset_name: str, base_dir: str = "experiments/ClipVIT"):
    """Devuelve (model_path, data_path) según la convención de carpetas."""
    if model_name.endswith("_gen"):
        task_name = model_name.replace("_gen", "")
        model_dir = f"clipvit-{task_name}"
        model_file = f"clipvit_{model_name}.pth"
    else:
        model_dir = f"clipvit-{model_name}"
        model_file = f"clipvit_{model_name}.pth"
    model_path = os.path.join(base_dir, "models", model_dir, model_file)

    if dataset_name.startswith("task_"):
        data_path = os.path.join(base_dir, "images_processed", dataset_name)
    else:
        data_path = os.path.join(base_dir, "images_processed", dataset_name, model_name)
    return model_path, data_path

# ════════════════════════════════════════════════════════════════════════════
# SINGLE EVALUATION ══════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_name: str, dataset_name: str, batch_size: int = 128):
    """Evaluate one checkpoint on one dataset."""
    model_path, data_path = construct_paths(model_name, dataset_name)

    if not os.path.exists(model_path):
        return {"error": f"Modelo no encontrado: {model_path}", "model": model_name, "dataset": dataset_name, "accuracy": None}
    if not os.path.exists(data_path):
        return {"error": f"Dataset no encontrado: {data_path}", "model": model_name, "dataset": dataset_name, "accuracy": None}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    img_size = processor.image_processor.size["shortest_edge"]
    normalize = transforms.Normalize(mean=processor.image_processor.image_mean,
                                     std=processor.image_processor.image_std)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    test_dir = os.path.join(data_path, "test")
    if not os.path.exists(test_dir):
        return {"error": f"Directorio de test no encontrado: {test_dir}", "model": model_name, "dataset": dataset_name, "accuracy": None}

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(test_dataset.classes)

    clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = ClipVisionClassifier(clip_base.vision_model, num_classes)
    
    # Load state dict and remove module prefix if present
    state_dict = torch.load(model_path, map_location=device)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"{model_name} → {dataset_name}", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total else 0.0
    return {"model": model_name, "dataset": dataset_name, "accuracy": acc, "correct": correct, "total": total, "error": None}

# ════════════════════════════════════════════════════════════════════════════
# GRID EVALUATION ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

def evaluate_multiple(models, datasets, batch_size: int = 128):
    combos = list(product(models, datasets))
    results = []
    for model_name, dataset_name in tqdm(combos, desc="Model × Dataset", unit="comb"):
        results.append(evaluate_model(model_name, dataset_name, batch_size))
    return results

# ════════════════════════════════════════════════════════════════════════════
# OUTPUT UTILITIES ═══════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# MAIN ═══════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="experiments/ClipVIT/outputs", help="Directorio para CSV y PNGs")
    args = parser.parse_args()

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

    print(f"🖥️ Dispositivo disponible: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print(f"🧠 GPUs visibles: {torch.cuda.device_count()}")

    results = evaluate_multiple(models_to_evaluate, datasets_to_evaluate, batch_size=args.batch_size)
    successful = collect_successful(results)

    if not successful:
        print("❌ No hay resultados exitosos para graficar o guardar.")
        return

    # Create pivot table: rows = dataset, cols = model, values = accuracy (%).
    df = (
        pd.DataFrame(successful)
        .pivot(index="dataset", columns="model", values="accuracy")
        * 100.0
    )

    out_dir = Path(args.out_dir)
    save_summary_csv(df, out_dir / "accuracy_summary.csv")
    save_csv_per_model(df, out_dir)
    plot_per_model(df, out_dir)


if __name__ == "__main__":
    main()

