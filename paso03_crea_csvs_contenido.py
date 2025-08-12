import time
import openai
import os
import yaml
import subprocess
import tiktoken
from pptx import Presentation
import subprocess
import base64
import requests
import json

ti = time.time()

path_entrada = f'PPTs_elementos'
path_slides = f'PPTs_separados'
path_salida = f'Informacion_Archivos'
path_prompts = f'Prompts'
path_originales = f'PPTs_originales'
nombres_archivos = os.popen(f'ls {path_entrada}').read().split()
#Para eliminar archivos temporales
nombres_archivos = [nombre for nombre in nombres_archivos if nombre[0]!='~'] 

# Cargar la configuración desde el archivo YAML
with open(f'config.yml', 'r') as file:
    config = yaml.safe_load(file)

unidad = config['extrae_ppt']['unidad']
conversiones = {'centímetros': 360000, 'pulgadas': 914400, 'pixeles': 9525}
velocidad_maxima = config['velocidad_maxima']


# Configura la API key
api_key =os.environ['OPENAI_API_KEY']
openai.api_key = api_key

def run_bash_command(command):
    try:
        subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error Obtenido: {e.stderr}")
    except Exception as e:
        print("Ocurrió un error inesperado:")
        print(str(e))


def guarda_json(nombre_json, datos_json):
    with open(nombre_json, 'w', encoding='utf-8') as file:
        json.dump(datos_json, file, ensure_ascii=False, indent=4) 

def imprime_mensaje(mje):
    largo = len(mje)
    print(' ')
    print('#'*(largo+10))
    print('###  '+mje+'  ###')
    print('#'*(largo+10))
    print(' ')


def guarda_prompt(prompt,nombre_archivo):
    with open(f'{path_prompts}/{nombre_archivo}', 'w') as file:
        file.write(prompt)

def calcula_tokens(st,modelo):

    encoder = tiktoken.encoding_for_model(modelo)  # Cambia por tu modelo si es necesario

    # Codifica el texto y cuenta los tokens
    tokens = encoder.encode(st)
    num_tokens = len(tokens)
    return num_tokens

def disminuye_velocidad(tokens_usados, tiempo_respuesta_minutos, velocidad_maxima=velocidad_maxima):
    # Fuerza al programa a esperar unos segundos si la velocidad de uso de tokens supera un
    # limite cercano al establecido por openai (actualmente 200,000 tokens por minuto)
    velocidad_actual = tokens_usados/tiempo_respuesta_minutos
    if velocidad_actual > velocidad_maxima:
        tiempo_espera_minutos = tokens_usados/velocidad_maxima - tiempo_respuesta_minutos
        time.sleep(tiempo_espera_minutos*60.)
    

def hacer_pregunta(pregunta, modelo = 'gpt-4o-mini'):
    t1 = time.time()
    response = openai.chat.completions.create(
        model=modelo, 
        messages=[
            {"role": "user", "content": pregunta},
        ]
    )
    # Obtener la respuesta
    respuesta = response.choices[0].message.content
    tokens_usados = calcula_tokens(pregunta,modelo)
    t2 = time.time()
    disminuye_velocidad(tokens_usados, (t2-t1)/60.)
    return respuesta, tokens_usados

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')


def hacer_pregunta_con_imagen(pregunta,path_imagen,modelo='gpt-4o-mini'):
    t1= time.time()
    # Getting the base64 string
    base64_image = encode_image(path_imagen)

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"}

    payload = {
      "model": modelo,
      "messages": [{"role": "user",
                    "content": [{"type": "text","text": pregunta},
                                {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
      "max_tokens": 1000}

    output = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    output_json = output.json()

    respuesta = output_json['choices'][0]['message']['content']
    tokens_usados = output_json['usage']['total_tokens']
    t2=time.time()
    disminuye_velocidad(tokens_usados, (t2-t1)/60.)

    return respuesta, output_json, tokens_usados


# Las consultas con este prompt están costando aproximadamente $0.01
def hace_prompt_resumen_preliminar_archivo(nombre_archivo):
    # A partir del nombre del archivo se crea un prompt para que Chat GPT entregue la descripcion
    #general del archivo que pueda ser usado como contexto para analizar cada diapositiva
    contexto = ('A continuación, se presentan descripciones detalladas de cada diapositiva de un '
                'archivo PPT. Tu tarea es usar esta información para crear un resumen general del '
                'archivo que servirá como contexto para interpretar cada diapositiva en pasos '
                'posteriores. El resumen debe capturar los elementos más importantes del archivo, '
                'alineados con los siguientes puntos:\n'
                '1. Tema Principal: Identifica y resume el tema principal o propósito del archivo, '
                'destacando de qué trata en términos generales.\n'
                '2. Información Útil para la Interpretación: Proporciona cualquier contexto '
                'relevante, como definiciones de acrónimos, términos técnicos, o conceptos '
                'clave que sean necesarios para entender correctamente el contenido del archivo.\n'
                '3. Flujo Semántico del Archivo: Describe brevemente la estructura y el flujo del '
                'archivo, señalando cómo se desarrolla el contenido a lo largo de las '
                'diapositivas. Identifica las principales secciones o etapas, como introducción, '
                'datos, resultados, etc.\n'
                '4. Objetivo del Archivo: Indica cuál es el objetivo del archivo, si es '
                'persuasivo, educativo, informativo, etc., para contextualizar mejor el '
                'contenido.\n'
                '5. Puntos Clave o Principales Mensajes: Enumera los mensajes o puntos clave que '
                'se destacan a lo largo del archivo.\n'
                '6. Información Visual Relevante: Describe los elementos visuales que tienen un '
                'significado específico en el archivo, como colores, formas, o íconos que son '
                'utilizados para representar acciones, notas, o mensajes. Pon especial atención '
                'a las primeras diapositivas que puedan contener explicaciones sobre estos '
                'elementos visuales.\n'
                '\n'
                'Usa toda la información de las descripciones de las diapositivas para construir '
                'este resumen. Evita repetir la misma información en diferentes secciones del '
                'resumen; en su lugar, trata de integrarla de manera cohesiva.\n'
                '\n')
    
    content_filenames = sorted(os.popen(f'ls {path_entrada}/{nombre_archivo}/slide*_content.txt')
                              .read().split())
    for filename in content_filenames:
        slide_number = int(filename.split('/')[-1].split('slide')[1].split('_content')[0])
        slide_file = open(filename,'r')
        slide_content = slide_file.read()
        slide_file.close()
        contexto += (f'\n[Descripción de la diapositiva numero {slide_number}]\n\n')
        contexto += (f'{slide_content}')
    return contexto


def hace_prompt_descripcion_slide(nombre_archivo, resumen_archivo,resumen_slides_anteriores,
                              descripcion_slide, numero_slide,total_slides):
    """
    A partir del resumen del archivo completo, la descripcion de los elementos de la slide,
    el resumen de las slides anteriores, la imagen de la slide, y el número de la slide c/r 
    a las slides totales se genera un prompt para preguntar a Chat GPT y pedirle que entregue 
    toda la información contenida en la slide.

    :param resumen_archivo: El resumen del archivo que sirve como contexto.
    :param descripcion_slide: Descripción detallada de los elementos presentes en la slide.
    :param numero_slide: El número de la slide dentro del archivo.
    :param total_slides: El número total de slides en el archivo.
    :return: Un string con el prompt a usar en la API de ChatGPT.
    """
    if resumen_slides_anteriores != '':
        texto_ultimos_inputs = (', además de un resumen del archivo, y un resumen de las '
                               'diapositivas anteriores a la actual.\n')
    else:
        texto_ultimos_inputs = (' y un resumen del archivo.\n')

    prompt = f"""
    Se te proporcionará la imagen de la diapositiva {numero_slide} de {total_slides} del archivo 
    \"{nombre_archivo}\" junto con la descripción de los elementos de texto e imagen de la slide
    {texto_ultimos_inputs}
    Tu tarea es describir todo el contenido que se muestra en la diapositiva, enfocándote en 
    explicar la información que se presenta en ella como lo haría un humano al interpretar la 
    diapositiva visualmente. Tu descripción debe ser lo suficientemente detallada como para que 
    alguien que lea la descripción pueda entender la diapositiva sin necesitar ver la imagen.

    [RESUMEN DEL ARCHIVO (Contexto General)]
    {resumen_archivo}

    """
    if resumen_slides_anteriores != '':
        prompt += f"""
    [RESUMEN DE LAS DIAPOSITIVAS ANTERIORES (Contexto General)]
    {resumen_slides_anteriores}

    """

    prompt += f"""
    [DESCRIPCIÓN DE LOS ELEMENTOS DE LA SLIDE {numero_slide}]
    {descripcion_slide}

    [INSTRUCCIONES]
    - Explica detalladamente el contenido de la diapositiva, considerando tanto el texto como 
    los elementos visuales descritos.
    - Enfócate en interpretar los mensajes y explicaciones, instrucciones, o cualquier tipo de 
    información relevante que se despliega en la diapositiva.
    - No te limites a enlistar los temas; describe las ideas y mensajes completos que la 
    diapositiva intenta comunicar, como lo haría un humano al explicar el contenido a otra persona.
    - Utiliza el contexto provisto del archivo, de los elementos de la slide actual y los 
    resumenes de las slides anteriores para asegurar una interpretación precisa.
    - Considera el uso de colores, símbolos y cualquier otra señal visual que pueda ser relevante 
    en la comprensión del mensaje de la diapositiva.
    - Recuerda que tendrás acceso a la imagen de la diapositiva, así que cualquier elemento visual 
    será interpretado para ayudarte en la descripción.

    Ejemplo de Respuesta Esperada:
    - Describe la diapositiva en un párrafo detallado, capturando las ideas centrales y cualquier 
    mensaje importante que se intenta comunicar, con énfasis en su aplicación práctica o en cómo 
    debería ser interpretado por el usuario final.

    Ahora, procede a analizar y describir la diapositiva {numero_slide} de {total_slides} del archivo, 
    basándote en la imagen y la información proporcionada.
    """
    return prompt

def hace_prompt_resumen_slide(archivo, numero_slide, total_slides, descripcion_completa_slide,
                              descripcion_automatica_slide):
    prompt = f"""
    Estás generando un resumen para la diapositiva {numero_slide} de {total_slides} del archivo "{archivo}".
    A partir de la descripción detallada proporcionada, el resumen debe centrarse en los objetivos, temas y el 
    propósito general de esta diapositiva, sin entrar en detalles específicos de los elementos visuales o 
    textuales presentes.
    El objetivo es que el resumen permita identificar la relevancia de la diapositiva en el contexto del archivo 
    completo para así ayudar a seleccionar las diapositivas necesarias para responder preguntas. 

    [Descripción detallada de la diapositiva]
    {descripcion_completa_slide}

    Si es útil, puedes utilizar la descripción técnica escrita abajo, que contiene los elementos de 
    texto e imagen de la diapositiva para mejorar la comprensión del contenido, pero mantén el resumen 
    enfocado en una vista general.

    [Descripción técnica de la diapositiva]
    {descripcion_automatica_slide}
    """
    return prompt


id_archivo = 0
id_pagina = 0 # Para tener un identificador global de las paginas
datos_archivos, datos_paginas = dict([]), dict([])
total_tokens = 0
for nombre_archivo in nombres_archivos:
    id_archivo += 1

    resumen_slides_anteriores = ''
    file_tokens = 0
    ti_archivo = time.time()

    imprime_mensaje(f'Analizando Archivo {id_archivo} de {len(nombres_archivos)}: "{nombre_archivo}"')
    print(f'Hasta ahora el tiempo transcurrido es {(ti_archivo-ti)/60:.2f} minutos, '
          f'y el gasto de tokens es {total_tokens:,}')
    run_bash_command(f'mkdir -p {path_salida}/Informacion_Paginas/{nombre_archivo}')
    ppt = Presentation(f'{path_originales}/{nombre_archivo}.pptx')
    n_slides = len(ppt.slides)
    num_ceros = len(str(n_slides))

    prompt_resumen_archivo = hace_prompt_resumen_preliminar_archivo(nombre_archivo)
    guarda_prompt(prompt_resumen_archivo,f'prompt_resume_archivo{id_archivo}.txt')
    respuesta_resumen_archivo,tokens_usados = hacer_pregunta(prompt_resumen_archivo)
    file_tokens += tokens_usados

    for n_slide in range(1,n_slides+1):
        id_pagina += 1
        print(f'Procesando Slide Número: {n_slide}')
        with open(f'{path_entrada}/{nombre_archivo}/slide{n_slide:03}_content.txt','r') as file:
            slide_content = file.read()
        imagen_path = f'{path_slides}/{nombre_archivo}/slide-{n_slide:0{num_ceros}}.png'
        prompt_describe_slide = hace_prompt_descripcion_slide(nombre_archivo,respuesta_resumen_archivo,
                                                              resumen_slides_anteriores,slide_content,
                                                              n_slide,n_slides)
        guarda_prompt(prompt_describe_slide,f'prompt_describe_archivo{id_archivo}_slide{n_slide}.txt')

        (respuesta_descripcion_slide,json_descripcion_resumen_slide,
         tokens_usados) = hacer_pregunta_con_imagen(prompt_describe_slide,imagen_path)
        file_tokens += tokens_usados

        with open(f'{path_salida}/Informacion_Paginas/{nombre_archivo}/'
                  f'Informacion_Slide{n_slide:03}.txt','w') as file:
            file.write(respuesta_descripcion_slide)        
        prompt_resume_slide = hace_prompt_resumen_slide(nombre_archivo,n_slide,n_slides,
                                                        respuesta_descripcion_slide,slide_content)
        guarda_prompt(prompt_resume_slide,f'prompt_resume_archivo{id_archivo}_slide{n_slide}.txt')
        respuesta_resumen_slide, tokens_usados = hacer_pregunta(prompt_resume_slide)
        file_tokens += tokens_usados
        datos_paginas[id_pagina] = {'ID Archivo': id_archivo, 'N_Pagina': n_slide,
                                    'Resumen': respuesta_resumen_slide}
        resumen_slides_anteriores += (f'\n- diapositiva{n_slide}: {respuesta_resumen_slide}\n')
    tiempo_archivo = f'{((time.time() - ti_archivo)/60.):.2f}'
    datos_archivos[id_archivo] = {'Nombre': nombre_archivo, 'Formato': 'pptx',
                                  'N_Paginas': n_slides, 'Total_Tokens': file_tokens,
                                  'Tiempo_Procesamiento_Minutos': tiempo_archivo,
                                  'Resumen': respuesta_resumen_archivo}
    total_tokens += file_tokens
    guarda_json(f'{path_salida}/archivos.json', datos_archivos)
    guarda_json(f'{path_salida}/paginas.json', datos_paginas)

tiempo_total = time.time() - ti
print(' ')
print(f'El script tomó {tiempo_total/60:.2f} minutos en analizar {len(nombres_archivos)} archivos')
print(f'Y se gastaron {total_tokens:,} tokens')
print(' ')


