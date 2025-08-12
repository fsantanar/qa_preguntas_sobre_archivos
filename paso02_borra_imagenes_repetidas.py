from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os
import time
import subprocess
import time
import json
ti = time.time()

path_entrada = f'data/PPTs_elementos'
nombres_archivos = os.popen(f'ls {path_entrada}').read().split() # Sin extension porque son las carpetas
#Para eliminar archivos temporales
nombres_archivos = [nombre for nombre in nombres_archivos if nombre[0]!='~'] 

def run_bash_command(command):
    try:
        subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error Obtenido: {e.stderr}")
    except Exception as e:
        print("Ocurrió un error inesperado:")
        print(str(e))


def comparar_imagenes_ssim(path1, path2):
    """
    Compara dos imágenes usando el índice de similitud estructural (SSIM).
    
    :param path1: Ruta de la primera imagen.
    :param path2: Ruta de la segunda imagen.
    :param umbral: Umbral de similitud para considerar las imágenes iguales (entre 0 y 1).
    :return: True si las imágenes son similares por encima del umbral, False en caso contrario.
    """
    # Cargar las imágenes y convertirlas a escala de grises para la comparación
    img1 = Image.open(path1).convert('L')
    img2 = Image.open(path2).convert('L')
    
    # Convertir las imágenes a arrays de numpy
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    # Calcular el índice de similitud estructural (SSIM)
    score, _ = ssim(img1_np, img2_np, full=True)

    return score

def agrupa_imagenes(carpeta_imagenes, umbral=0.98):
    
    nombres_archivos = os.popen(f'ls {carpeta_imagenes}').read().split('\n')[:-1]
    nombres_imagenes = [nombre for nombre in nombres_archivos
                        if ('.' in nombre and nombre.split('.')[-1] in ['png','jpg','jpeg','pdf','wmf'])]

    grupos_de_imagenes = dict([])

    for nombre in nombres_imagenes:
        # Bandera para controlar si la imagen fue agregada a un grupo
        asignada_a_grupo = False
        
        for nombre_grupo, imagenes_grupo in grupos_de_imagenes.items():
            # Compara con la primera imagen del grupo
            imagen_comparacion = imagenes_grupo[0]
            try:
                puntaje = comparar_imagenes_ssim(f'{carpeta_imagenes}/{nombre}',
                                                 f'{carpeta_imagenes}/{imagen_comparacion}')
            except ValueError:
                puntaje = -1
            

            if puntaje > umbral:
                grupos_de_imagenes[nombre_grupo].append(nombre)
                asignada_a_grupo = True
                break
        
        # Si la imagen no fue asignada a ningún grupo, crea uno nuevo
        if not asignada_a_grupo:
            grupos_de_imagenes[nombre] = [nombre]
    
    imagenes_a_borrar = dict([])
    for nombre_grupo, imagenes_grupo in grupos_de_imagenes.items():
        for imagen in imagenes_grupo:
            if imagen!=nombre_grupo:
                imagenes_a_borrar[imagen]=nombre_grupo


    return imagenes_a_borrar


n_archivo = 0
for archivo in nombres_archivos:
    n_archivo += 1
    dt = time.time() - ti
    print(' ')
    print(f'Analizando archivo {n_archivo} de {len(nombres_archivos)}: {archivo}')
    print(f'Han transcurrido {dt/60:.2f} minutos de ejecución')
    imagenes_a_borrar = agrupa_imagenes(f'{path_entrada}/{archivo}')
    with open(f'{path_entrada}/{archivo}/imagenes_borradas.json', 'w') as archivo_json:
        json.dump(imagenes_a_borrar, archivo_json)
    for imagen in imagenes_a_borrar.keys():
        run_bash_command(f'rm {path_entrada}/{archivo}/{imagen}')

tf = time.time()
t_total = tf-ti
print(' ')
print(f'El script completo demoró {t_total/60:.2f} minutos en revisar {len(nombres_archivos)} archivos')
print(' ')