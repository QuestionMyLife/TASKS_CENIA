import os
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image

def reorganize_and_rename_all_tasks(base_dir, output_dir, split_names, valid_task_dirs):
    # Inicializamos los contadores para las clases
    class_0_counter = 1  # Clase 0
    class_1_counter = 1  # Clase 1

    # Recorremos las carpetas de tipo (normal, normal_small_radius, ...) especificadas
    for task_type in os.listdir(base_dir):
        # Solo procesamos las carpetas que están en la lista valid_task_dirs
        if task_type not in valid_task_dirs:
            continue  # Si no está en valid_task_dirs, saltarlo

        task_type_path = os.path.join(base_dir, task_type)
        if not os.path.isdir(task_type_path):
            continue  # Ignoramos si no es una carpeta

        # Recorremos todas las tasks dentro de cada carpeta TYPE
        for task_dir in os.listdir(task_type_path):
            task_path = os.path.join(task_type_path, task_dir)
            if not os.path.isdir(task_path):
                continue  # Ignoramos si no es una carpeta

            print(f"Procesando task: {task_dir} en {task_type}")

            # Ahora creamos una nueva carpeta para el dataset combinado
            output_task_dir = os.path.join(output_dir, f"{task_dir}_gen")
            os.makedirs(output_task_dir, exist_ok=True)

            # Recorremos los splits (train, val, test)
            for split in split_names:
                output_split_dir = os.path.join(output_task_dir, split)
                os.makedirs(output_split_dir, exist_ok=True)

                # Recorremos las carpetas que contienen imágenes de ese task
                task_subdir_path = os.path.join(task_type_path, task_dir, split)
                print(f"🔍 Buscando en: {task_subdir_path}")  # Depuración
                if not os.path.exists(task_subdir_path):
                    print(f"⚠️ Carpeta no encontrada: {task_subdir_path}")
                    continue  # Si no existe la carpeta, saltar

                # Obtenemos todas las imágenes .png en esa carpeta
                images = glob(os.path.join(task_subdir_path, "*.png"))
                print(f"📂 Imágenes encontradas en {task_subdir_path}: {len(images)}")

                if len(images) == 0:
                    print(f"⚠️ No se encontraron imágenes en {task_subdir_path}.")

                for path in tqdm(images, desc=f"Combinando {split} en {task_dir}"):
                    filename = os.path.basename(path)
                    try:
                        # Asignar clase basada en el prefijo
                        if "class_0" in filename:  # Clase 0
                            class_id = 0
                            # Asegurarse de crear la carpeta class_0 dentro de la carpeta train, val, test
                            class_dir = os.path.join(output_split_dir, "class_0")
                            os.makedirs(class_dir, exist_ok=True)  # Crear la carpeta de clase si no existe
                            new_filename = f"class_0_{class_0_counter:06d}.jpg"
                            class_0_counter += 1
                        else:  # Clase 1
                            class_id = 1
                            # Asegurarse de crear la carpeta class_1 dentro de la carpeta train, val, test
                            class_dir = os.path.join(output_split_dir, "class_1")
                            os.makedirs(class_dir, exist_ok=True)  # Crear la carpeta de clase si no existe
                            new_filename = f"class_1_{class_1_counter:06d}.jpg"
                            class_1_counter += 1

                        # Guardamos la imagen convertida a JPG en la carpeta correspondiente
                        new_path = os.path.join(class_dir, new_filename)

                        # Convertir a JPG y guardar con el nuevo nombre
                        with Image.open(path) as img:
                            img = img.convert("RGB")
                            img.save(new_path, format="JPEG", quality=95)

                    except Exception as e:
                        print(f"⚠️ Error con archivo {filename}: {e}")

    print(f"\n✅ Dataset combinado y reestructurado en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combinación de datasets para todas las tasks desde múltiples carpetas y renombrado de imágenes.")
    parser.add_argument("--base_dir", type=str, required=True, help="Ruta al directorio base que contiene las carpetas de tipo 'normal', 'normal_small_radius', etc.")
    parser.add_argument("--output_dir", type=str, required=True, help="Ruta destino para el dataset combinado.")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Nombres de los splits separados por coma.")
    parser.add_argument("--valid_task_dirs", type=str, default="normal,normal_small_radius,smooth", help="Nombres de las carpetas de datasets a procesar, separados por coma (ej. 'normal,normal_small_radius').")

    args = parser.parse_args()
    split_names = args.splits.split(",")
    valid_task_dirs = args.valid_task_dirs.split(",")  # Convertir a lista

    reorganize_and_rename_all_tasks(args.base_dir, args.output_dir, split_names, valid_task_dirs)
