import subprocess
import yaml
import time

# Cargar la configuración desde el archivo YAML
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

nombre_db = config['database']['name']

def crear_base_datos(nombre_db):
    # Solicitar confirmación del usuario
    print(f"Estás a punto de borrar la base de datos {nombre_db}")
    print(f"Tienes 5 segundos para cancelar la operación... (Ctrl+C)")
    try:
        # Dar al usuario un momento para cancelar la operación
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
        return

    # Borrar la base de datos si existe
    subprocess.run(['psql', '-U', 'felipesantana', '-d', 'postgres', '-c',
                    f"DROP DATABASE IF EXISTS {nombre_db};"],
                    check=False)

    # Intentar crear la base de datos
    try:
        resultado = subprocess.run(['psql', '-U', 'felipesantana', '-d', 'postgres', '-c',
                                    f"CREATE DATABASE {nombre_db};"],
                                    text=True, capture_output=True, check=False)
        if "already exists" in resultado.stderr:
            print("Hubo un error: la base de datos aun existe después de intentar borrarla.")
        else:
            print(f"La base de datos '{nombre_db}' fue creada exitosamente.")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e}")

crear_base_datos(nombre_db)
