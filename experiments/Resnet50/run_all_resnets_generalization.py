import os
import subprocess

def run_prepare_data(original_dir, output_dir, valid_task_dirs):
    print(f"🔧 Reorganizando dataset desde {original_dir}")
    subprocess.run([
        "python", "experiments/Resnet50/prepare_data_generalization.py",  # Cambié el nombre del script a prepare_data_generalization.py
        "--base_dir", original_dir,
        "--output_dir", output_dir,
        "--splits", "train,val,test",
        "--valid_task_dirs", ",".join(valid_task_dirs)  # Pasamos las carpetas a procesar
    ], check=True)
    print(f"✅ Dataset procesado en: {output_dir}\n")


def run_training(data_dir, output_dir, cuda_devices, batch_size, lr, epochs):
    print(f"🚀 Entrenando modelo con datos en: {data_dir}")
    command = [
        "python", "experiments/Resnet50/Resnet50_training.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--cuda_devices", cuda_devices,
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--epochs", str(epochs)
    ]
    subprocess.run(command, check=True)
    print(f"✅ Entrenamiento completado y modelo guardado en: {output_dir}\n")


if __name__ == "__main__":
    BASE_INPUT = "/workspace1/gonzalo.fuentes/CVR_CENIA/experiments/images"
    BASE_OUTPUT_ROOT = "/workspace1/gonzalo.fuentes/CVR_CENIA/experiments/Resnet50/images_processed"
    BASE_MODELS = "/workspace1/gonzalo.fuentes/CVR_CENIA/experiments/Resnet50/models"

    test_task_dirs = [
        "normal", "normal_small_radius", "smooth_low_fourier", "rigid_triangles_color", "symm_color_big",
        "rigid_decagons_color", "rigid_hexagons", "rigid_rectangles", "smooth_high_fourier", "smooth_mid_fourier_color", "symm_short",
        "irregular_decagons_color", "irregular_hexagons","irregular_pentagons_color"
    ]

    # for test in test_task_dirs:
    #     run_prepare_data(
    #         BASE_INPUT,
    #         os.path.join(BASE_OUTPUT_ROOT, f"{test}"),
    #         [test]      
    #     )
    
    cuda_devices = "0,1"

    # valid_task_dirs = ["normal", "irregular_hexagons", "smooth_low_fourier", "symm_color_big", "rigid_triangles_color"]
    valid_task_dirs = ["normal"]

    tasks = [
        # {
        #     "name": "task_MTS",
        #     "batch_size": 128,
        #     "lr": 5e-6,
        #     "epochs": 20
        # },
        # {
        #     "name": "task_SD",
        #     "batch_size": 256,
        #     "lr": 1e-4,
        #     "epochs": 20
        # },
        {
            "name": "task_SOSD",
            "batch_size": 64,
            "lr": 5e-5,
            "epochs": 20
        },
        {
            "name": "task_RMTS",
            "batch_size": 256,
            "lr": 5e-5,
            "epochs": 20
        }
    ]

    # # Llamar a la función que reorganiza los datos, pasando las carpetas válidas
    # # run_prepare_data(BASE_INPUT, BASE_OUTPUT_ROOT, valid_task_dirs)

    # Tareas sin simetria
    for task in tasks:
        name = task["name"]
        print(f"\n========================= 🔁 Procesando {name} =========================")

        # 🛠️ CORRECCIÓN: usamos BASE_OUTPUT_ROOT directamente como carpeta de salida del prepare_data
        processed_dir = os.path.join(BASE_OUTPUT_ROOT, f"{name}_gen")

        model_output_dir = os.path.join(BASE_MODELS, f"resnet50-{name}")

        # Ahora usamos processed_dir (con _gen) para entrenar el modelo
        run_training(
            data_dir=processed_dir,  # Usamos las imágenes procesadas con _gen
            output_dir=model_output_dir,
            cuda_devices=cuda_devices,
            batch_size=task["batch_size"],
            lr=task["lr"],
            epochs=task["epochs"]
        )

    # tasks = [
    #     {
    #         "name": "task_sym_MTS",
    #         "batch_size": 128,
    #         "lr": 5e-5,
    #         "epochs": 20
    #     },
    #     {
    #         "name": "task_sym_SD",
    #         "batch_size": 128,
    #         "lr": 5e-6,
    #         "epochs": 20
    #     },
    #     {
    #         "name": "task_sym_classification",
    #         "batch_size": 128,
    #         "lr": 5e-5,
    #         "epochs": 20
    #     },
    # ]

    # # Tareas de simetria
    # # valid_task_dirs = ["normal", "irregular_pentagons_color", "smooth_mid_fourier_color", "normal_small_radius"]
    # valid_task_dirs = ["normal"]
    # # run_prepare_data(BASE_INPUT, BASE_OUTPUT_ROOT, valid_task_dirs)

    # for task in tasks:
    #     name = task["name"]
    #     print(f"\n========================= 🔁 Procesando {name} =========================")

    #     # 🛠️ CORRECCIÓN: usamos BASE_OUTPUT_ROOT directamente como carpeta de salida del prepare_data
    #     processed_dir = os.path.join(BASE_OUTPUT_ROOT, f"{name}_gen")

    #     model_output_dir = os.path.join(BASE_MODELS, f"resnet50-{name}")

    #     # Ahora usamos processed_dir (con _gen) para entrenar el modelo
    #     run_training(
    #         data_dir=processed_dir,  # Usamos las imágenes procesadas con _gen
    #         output_dir=model_output_dir,
    #         cuda_devices=cuda_devices,
    #         batch_size=task["batch_size"],
    #         lr=task["lr"],
    #         epochs=task["epochs"]
    #     )
