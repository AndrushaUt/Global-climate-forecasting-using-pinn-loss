import os
import gcsfs
from tqdm import tqdm

def download_weatherbench2(gcs_path, local_dir="./weatherbench2_data", overwrite=False):
    """
    Скачивает датасет WeatherBench2 из указанного пути GCS локально.
    
    Параметры:
        gcs_path (str): Путь в Google Cloud Storage (например, 'weatherbench2/era5/1.0/t2m')
        local_dir (str): Локальная директория для сохранения
        overwrite (bool): Перезаписывать ли существующие файлы
    """
    # Инициализация файловой системы GCS
    fs = gcsfs.GCSFileSystem(project='public')
    
    # Проверяем существование пути
    if not fs.exists(gcs_path):
        print(f"Ошибка: Путь {gcs_path} не существует в Google Cloud Storage")
        return
    
    # Получаем список всех файлов для скачивания
    if fs.isdir(gcs_path):
        print(f"Сканирование директории {gcs_path}...")
        files = fs.find(gcs_path)
        # Фильтруем только файлы (не директории)
        files = [f for f in files if not fs.isdir(f)]
    else:
        # Если указан один файл
        files = [gcs_path]
    
    print(f"Найдено {len(files)} файлов для скачивания")
    
    # Создаем локальную директорию, если не существует
    os.makedirs(local_dir, exist_ok=True)
    
    # Скачиваем каждый файл с прогресс-баром
    downloaded = 0
    skipped = 0
    failed = 0
    
    for file_path in tqdm(files, desc="Скачивание файлов"):
        # Определяем локальный путь для файла
        rel_path = file_path.replace("weatherbench2/", "")
        local_path = os.path.join(local_dir, rel_path)
        
        # Создаем директории в локальном пути, если не существуют
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Пропускаем существующие файлы, если не указан overwrite
        if os.path.exists(local_path) and not overwrite:
            skipped += 1
            continue
        
        try:
            # Скачиваем файл
            fs.get(file_path, local_path)
            downloaded += 1
        except Exception as e:
            print(f"Ошибка при скачивании {file_path}: {e}")
            failed += 1
    
    print(f"\nЗагрузка завершена!")
    print(f"Скачано: {downloaded} файлов")
    print(f"Пропущено (уже существует): {skipped} файлов")
    print(f"Ошибки: {failed} файлов")

# Пример использования
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python script.py <путь_в_gcs> [локальная_директория]")
        print("Например: python script.py weatherbench2/era5/1.0/t2m ./my_data")
        
        # Показываем доступные пути
        print("\nДоступные пути:")
        fs = gcsfs.GCSFileSystem(project='public')
        for item in fs.ls('weatherbench2/era5'):
            print(f" - {item}")
        sys.exit(1)
    
    # Получаем аргументы
    gcs_path = sys.argv[1]
    local_dir = sys.argv[2] if len(sys.argv) > 2 else "./weatherbench2_data"
    
    # Добавляем префикс weatherbench2/, если пользователь его не указал
    if not gcs_path.startswith("weatherbench2/"):
        gcs_path = f"weatherbench2/{gcs_path}"
    
    # Скачиваем данные
    download_weatherbench2(gcs_path, local_dir)
