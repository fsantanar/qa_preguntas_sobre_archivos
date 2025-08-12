import time
import psycopg2
import yaml
import json

def imprime_mensaje(mje):
    largo = len(mje)
    print(' ')
    print('#'*(largo+10))
    print('###  '+mje+'  ###')
    print('#'*(largo+10))
    print(' ')

def lee_json(nombre_archivo_json):
    with open(nombre_archivo_json, 'r') as file:
        data = json.load(file)
    return data

# Función para insertar datos en la tabla
def insertar_datos(conn, nombre_tabla, diccionario_datos, columnas):
    cursor = conn.cursor()

    # Crear la parte dinámica de placeholders (%s) según la cantidad de columnas
    placeholders = ', '.join(['%s'] * (len(columnas) + 1))  # +1 para incluir el id

    # Crear la consulta SQL con los placeholders
    query = f"INSERT INTO {nombre_tabla} VALUES ({placeholders})"

    # Iterar sobre el JSON e insertar los valores en la tabla
    try:
        for key, value in diccionario_datos.items():
            # Preparar los valores en el orden correcto basado en las columnas proporcionadas
            valores = [value[col] for col in columnas]

            # Incluir la llave principal (id) como el primer valor
            valores.insert(0, key)
            
            # Ejecutar la inserción con los valores
            cursor.execute(query, valores)
    
        # Guardar los cambios en la base de datos
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error durante la transacción: {e}")
        conn.rollback()  # Deshacer la transacción si ocurre un error
    finally:
        cursor.close()

def crear_tabla(conn, nombre_tabla, comando):
    cursor = conn.cursor()
    imprime_mensaje(f'Creando la tabla {nombre_tabla}...')
    cursor.execute(comando)
    conn.commit()



# Cargar la configuración desde el archivo YAML
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

db_name = config['database']['name']
super_usuario_db = config['database']['super_usuario']
path_entrada = f'Informacion_Archivos'

# Conectar a la base de datos PostgreSQL
conn = psycopg2.connect(
    host="localhost",  # Reemplaza por tu host si es diferente
    database=db_name,  # Nombre de la base de datos
    user=super_usuario_db  # Usuario de PostgreSQL
)

cursor = conn.cursor()

# Comando para crear la tabla archivos si no existe
crear_tabla_archivos = """
CREATE TABLE IF NOT EXISTS archivos (
    id INT PRIMARY KEY,
    nombre VARCHAR(255),
    formato VARCHAR(15),
    paginas INT,
    resumen TEXT
);
"""

# Comando para crear la tabla paginas si no existe
crear_tabla_paginas = """
CREATE TABLE IF NOT EXISTS paginas (
    id INT PRIMARY KEY,
    archivo INT,
    n_pagina INT,
    descripcion TEXT
);
"""

# Comando para crear la tabla respuestas
crear_tabla_respuestas = """
CREATE TABLE IF NOT EXISTS respuestas (
  id INTEGER PRIMARY KEY,
  pregunta TEXT,
  archivo INTEGER,
  pagina_inicial INTEGER,
  pagina_final INTEGER,
  id_pregunta_mismo_tema INTEGER,
  frecuencia INTEGER,
  tokens_estrategia INTEGER,
  tokens_contexto_archivo INTEGER,
  tokens_contexto_paginas INTEGER,
  tokens_respuesta INTEGER,
  segundos_estrategia_total INTEGER,
  segundos_contexto_archivo INT,
  segundos_contexto_paginas INT,
  segundos_respuesta INTEGER,
  segundos_estrategia_ultima INTEGER,
  comparaciones_estrategia_ultima INT,
  momento_primera_respuesta TIMESTAMP,
  momento_ultima_respuesta TIMESTAMP,
  version_codigo VARCHAR(20),
  nota_humano FLOAT,
  respuesta TEXT
);
"""

# Comando para crear la tabla fallos
crear_tabla_fallos = """
CREATE TABLE IF NOT EXISTS fallos (
    id SERIAL PRIMARY KEY,
    pregunta TEXT,
    momento_respuesta TIMESTAMP,
    version_codigo VARCHAR(20),
    exito_contexto_archivo BOOLEAN,
    exito_contexto_paginas BOOLEAN,
    archivo TEXT,
    pagina_inicial INT,
    pagina_final INT,
    tokens_estrategia INT,
    tokens_contexto_archivo INT,
    tokens_contexto_paginas INT,
    tokens_respuesta INT,
    segundos_estrategia INT,
    segundos_contexto_archivo INT,
    segundos_contexto_paginas INT,
    segundos_respuesta INT
);
"""

# Ejecutar la creación de las tablas
crear_tabla(conn, 'archivos', crear_tabla_archivos)
crear_tabla(conn, 'paginas', crear_tabla_paginas)
crear_tabla(conn, 'respuestas', crear_tabla_respuestas)
crear_tabla(conn, 'fallos', crear_tabla_fallos)



# Leer datos de los JSON
datos_archivos = lee_json(f'{path_entrada}/archivos.json')
columnas_archivos = ['Nombre', 'Formato', 'N_Paginas', 'Resumen']
datos_paginas = lee_json(f'{path_entrada}/paginas.json')
columnas_paginas = ['ID Archivo', 'N_Pagina', 'Resumen']


# Llenar las tablas archivos y paginas
imprime_mensaje("Llenando la tabla archivos...")
insertar_datos(conn, 'archivos', datos_archivos, columnas_archivos)
imprime_mensaje("Llenando la tabla paginas...")
insertar_datos(conn, 'paginas', datos_paginas, columnas_paginas)


conn.close()
