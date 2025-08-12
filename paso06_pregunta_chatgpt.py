
import time
ti=time.time()
def imprime_tiempo(mensaje):
    dt = time.time()-ti
    print(f'Estado: {mensaje}, Time Covered: {dt:.2f}')
imprime_tiempo('Inicio')
import openai
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from peewee import Model, IntegerField, TextField, CharField, PostgresqlDatabase, FloatField
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging
logging.set_verbosity_error()
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import warnings
imprime_tiempo('Cargados los Módulos')

# Suprimir el FutureWarning sobre clean_up_tokenization_spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")





### IMPORTANTE:
#
# IMPLEMENTAR UN SISTEMA DE TAGS DE PALABRAS O CONCEPTOS QUE SE DEFINAN POR ARCHIVO Y/O POR PAGINA
#

# Cargar la configuración desde el archivo YAML
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
db_name = config['database']['name']
super_usuario_db = config['database']['super_usuario']
super_usuario_pw_db = config['database']['super_usuario_pw']


path_archivos = f'data/Informacion_Archivos/Informacion_Paginas'
maximo_paginas = config['maximo_paginas']
umbral_semejanza = config['decide_estrategia']['umbral_semejanza']
umbral_implicancia = config['decide_estrategia']['umbral_implicancia']

# Configurar la conexión a la base de datos PostgreSQL
db = PostgresqlDatabase(db_name, user=super_usuario_db, password=super_usuario_pw_db, host='localhost')


# Definir los modelos de Peewee correspondientes a las tablas
class BaseModel(Model):
    class Meta:
        database = db

class Archivos(BaseModel):
    id = IntegerField(primary_key=True)  # Define el campo id como la clave primaria
    nombre = CharField(max_length=255, null=True)  # Campo varchar(255)
    formato = CharField(max_length=15, null=True)  # Campo varchar(15)
    paginas = IntegerField(null=True)  # Campo integer
    resumen = TextField(null=True)  # Campo text

    class Meta:
        table_name = 'archivos'

class Paginas(BaseModel):
    id = IntegerField()
    archivo = IntegerField()
    n_pagina = IntegerField()
    descripcion = TextField()

    class Meta:
        table_name = 'paginas'


class Respuestas(BaseModel):
    id = IntegerField()
    pregunta = TextField()
    archivo = TextField()
    pagina_inicial = IntegerField()
    pagina_final = IntegerField()
    frecuencia = IntegerField()
    nota_humano = FloatField()
    respuesta = TextField()
    class Meta:
        table_name = 'respuestas'

class Fallos(BaseModel):
    id = IntegerField()
    pregunta = TextField()
    archivo = TextField()
    pagina_inicial = IntegerField()
    pagina_final = IntegerField()
    frecuencia = IntegerField()
    class Meta:
        table_name = 'fallos'



query_archivos = (Archivos.select())
df_archivos = pd.DataFrame(list(query_archivos.dicts()))

query_paginas = (Paginas.select())
df_paginas = pd.DataFrame(list(query_paginas.dicts()))

query_respuestas = (Respuestas.select())
df_respuestas = pd.DataFrame(list(query_respuestas.dicts()))
if len(df_respuestas)==0:
    max_actual_id_respuestas=0
else:
    max_actual_id_respuestas = max(df_respuestas['id'])

query_fallos = (Fallos.select())
df_fallos = pd.DataFrame(list(query_fallos.dicts()))
if len(df_fallos)==0:
    max_actual_id_fallos=0
else:
    max_actual_id_fallos = max(df_fallos['id'])

# Reemplaza con tu API key de OpenAI
api_key =os.environ['OPENAI_API_KEY']

# Configura la API key
openai.api_key = api_key
imprime_tiempo('Cargadas las tablas, Antes de cargar modelos')
# Modelo de semejanza de frases
modelo_semejanza = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
imprime_tiempo('Definido el modelo de Semejanza')
# Modelo de implicancia semántica
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
modelo_implicancia = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
imprime_tiempo('Definido el modelo de Implicancia')

def son_semejantes(frase1, frase2, umbral=0.8):
    # Genera embeddings para las frases
    embedding1 = modelo_semejanza.encode(frase1, convert_to_tensor=True)
    embedding2 = modelo_semejanza.encode(frase2, convert_to_tensor=True)

    # Calcula la similitud coseno
    semejanza = util.cos_sim(embedding1, embedding2).item()

    # Retorna True si la similitud es mayor al umbral
    return semejanza >= umbral, semejanza

def son_implicantes(frase1, frase2, umbral=0.8):
    # Función auxiliar para calcular la implicancia en un sentido
    def calcular_implicancia(f1, f2):
        # Preparar las entradas para el modelo
        inputs = tokenizer.encode_plus(f1, f2, return_tensors='pt', truncation=True)
        
        # Realizar la predicción
        outputs = modelo_implicancia(**inputs)
        logits = outputs.logits
        
        # Los índices 0, 1, 2 corresponden a contradicción, neutral y implicación
        probs = torch.softmax(logits, dim=1)
        contradiction_prob, neutral_prob, entailment_prob = probs[0].tolist()
        
        return entailment_prob, contradiction_prob

    # Calcular implicancia en ambas direcciones
    entailment_prob_1, contradiction_prob_1 = calcular_implicancia(frase1, frase2)
    entailment_prob_2, contradiction_prob_2 = calcular_implicancia(frase2, frase1)

    # Elegir el mayor valor de implicancia
    if entailment_prob_1 >= entailment_prob_2:
        entailment_prob_final = entailment_prob_1
        contradiction_prob_final = contradiction_prob_1
    else:
        entailment_prob_final = entailment_prob_2
        contradiction_prob_final = contradiction_prob_2

    # Evaluar si el modelo considera las frases como implicadas
    return entailment_prob_final >= umbral, entailment_prob_final, contradiction_prob_final



def hacer_pregunta(pregunta):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "user", "content": pregunta},
            ]
        )
        # Obtener la respuesta
        respuesta = response.choices[0].message.content
        return respuesta
    except Exception as e:
        return f"Error al hacer la pregunta: {e}"

def hace_prompt_elige_archivo(pregunta,dataframe_archivos):
    # A partir de la pregunta original del usuario y el dataframe con la info de los archivos
    # Crea el texto de la pregunta para elegir el archivo necesario para responder la pregunta
    # del usuario
    prompt = (f'Necesito eventualmente responder la pregunta "{pregunta}" usando la información '
              f'contenida en un set de archivos. A continuación te muestro la id, el nombre y un '
              f'resumen de cada archivo. Quiero que me respondas solo con la id del archivo más '
              f'útil para responder la pregunta (y si ninguno sirve que respondas "-1")\n')
    for __,row in dataframe_archivos.iterrows():
        prompt += (f'id={row["id"]}, nombre={row["nombre"]}, resumen={row["resumen"]}\n')
    return prompt

def hace_prompt_elige_paginas(pregunta, dataframe_archivos, dataframe_paginas_archivo):
    # A partir de la pregunta original del usuario y el dataframe con las paginas del archivo
    # que eligio chat gpt como util para la respuesta, se crea el texto para pedirle que eliga
    # el rango de paginas necesario para responder la pregunta
    nombre_archivo = dataframe_archivos[dataframe_archivos['id']==
                                        dataframe_paginas_archivo['archivo'].values[0]]['nombre'].values[0]
    prompt = (f'Necesito identificar el rango de páginas del archivo "{nombre_archivo}" necesario '
              f'para responder la pregunta: "{pregunta}". Para eso a continuación te indico para cada '
              f'página un resumen de su contenido.\n\n')
    for __,row in dataframe_paginas_archivo.iterrows():
        prompt += (f'Página {row["n_pagina"]}: "{row["descripcion"]}"\n')
    prompt += (f'\nBasado en esta información, responde exclusivamente con el rango de páginas que '
               f'son más útiles para responder la pregunta: "{pregunta}". Si no hay páginas útiles, '
               f'responde solo con "-1". La respuesta debe ser solo el rango de páginas, '
               f'por ejemplo, "1-1".')

    return prompt  

def hace_prompt_final(pregunta, archivo_elegido, pagina_inicial, pagina_final):
    # A partir de la pregunta original del usuario, el archivo y el tema seleccionados
    # por chat gpt como utiles para la respuesta se crea el texto para obtener la respuesta final

    contexto = (f'Necesito responder la pregunta "{pregunta}" usando la información '
                f'contenida en un archivo llamado "{archivo_elegido}" que trata de distintos temas. '
                f'Dentro de él se determinó el rango de páginas más útiles para responder la pregunta. '
                f'Por tanto abajo te pego la descripción detallada del contenido de cada página en '
                f'ese rango de ese rango de páginas para que las uses para responder la pregunta '
                f'del usuario.\n\n')

    contenido_paginas=''
    for pagina in range(pagina_inicial,pagina_final+1):
        path_archivo = f'{path_archivos}/{archivo_elegido}/Informacion_Slide{pagina:03}.txt'
        archivo_actual = open(path_archivo,'r')
        contenido_pagina = archivo_actual.read()
        archivo_actual.close()
        contenido_paginas += contenido_pagina
    
    prompt = contexto + contenido_paginas
    return prompt

def determina_nombre_archivo_elegido(respuesta_chatgpt,dataframe_archivos):
    found, archivo=False,''
    if not respuesta_chatgpt.isdigit():
        print(f'La respuesta de ChatGPT ({respuesta_chatgpt}) sobre el archivo '
              f'a usar no fue entregada como número')
    elif int(respuesta_chatgpt) not in dataframe_archivos['id'].values:
        print('El número de archivo entregado por Chat GPT no corresponde a una ID válida')
    else:
        found=True
        archivo=dataframe_archivos[dataframe_archivos["id"]==int(respuesta_chatgpt)]["nombre"].values[0]
        print(f'El archivo necesario para responder la pregunta según Chat GPT es {archivo}')
    return found,archivo

def determina_paginas_elegidas(respuesta_chatgpt,n_paginas):
    # Esta función determina que se hace con la respuesta de Chat GPT de las paginas elegidas
    # Si la respuesta es valida se intenta responder la pregunta con ese rango de paginas,
    # si no es valida se intenta usar el archivo completo.
    # Si el rango a intentar no es demasiado largo se hace la pregunta a Chat GPT y por tanto
    # respond es True, pero si el rango a intentar es demasiado largo no se hace la pregunta
    # para ahorrar tokens.
    found, respond, pagina_inicial, pagina_final = False, False, -1, -1
    respuesta_sep = respuesta_chatgpt.split('-')
    if len(respuesta_sep)!=2 or not respuesta_sep[0].isdigit() or not respuesta_sep[1].isdigit():
        print(f'La respuesta de ChatGPT ({respuesta_chatgpt}) no corresponde a un rango de números')
    elif ((int(respuesta_sep[0])<=0 or int(respuesta_sep[0])>n_paginas)
          or (int(respuesta_sep[0])<=0 or int(respuesta_sep[0])>n_paginas)):
        print(f'El rango de página entregado por Chat GPT ({respuesta_chatgpt}) no existe en el archivo')
    else:
        found=True
        pagina_inicial, pagina_final = int(respuesta_sep[0]), int(respuesta_sep[1])
        print('El rango de páginas necesario para responder la pregunta según Chat GPT es '
              f'de la {int(respuesta_sep[0])} a la {int(respuesta_sep[1])}')
        
    # Ahora determinamos si se va a hacer la pregunta final o no
    # dependiendo del largo en paginas que deberiamos usar de input
    if found is False: # Si no se encuentra el rango de paginas más util se intenta el archivo entero
        pagina_inicial, pagina_final=1, n_paginas
        print('Como Chat GPT no pudo encontrar el rango de páginas más utiles intentaremos '
              'usar el archivo completo')

    n_paginas_input = 1 + pagina_final - pagina_inicial

    if n_paginas_input <= maximo_paginas:
        respond=True
        print(f'Como el rango de páginas a usar es corto lo usaremos como input')
    else:
        print(f'Como el rango de páginas a usar es demsiado largo no se puede responder la pregunta')
    
    return found, respond, pagina_inicial, pagina_final

def determina_estrategia(pregunta,dataframe_respuestas, umbral_semejanza=umbral_semejanza,
                         umbral_implicancia=umbral_implicancia):
    # Esta función revisa la pregunta actual y las ya hechas representadas en el dataframe
    # Y con eso decide si hay se puede usar una respuesta existente, si se puede usar un rango
    # de paginas existente, o si hay que determinar el rango de paginas a usar para la nueva
    # pregunta a chat gpt

    estrategia, indice = 'pregunta_nueva', -1
    if len(dataframe_respuestas)==0:
        return estrategia, indice
    
    indices, semejanzas, implicancias = [], [], []
    for ind,row in dataframe_respuestas.iterrows():
        pregunta_comparacion = row['pregunta']
        implicancia = son_implicantes(pregunta, pregunta_comparacion)[1]
        semejanza = son_semejantes(pregunta, pregunta_comparacion)[1]
        indices.append(ind)
        semejanzas.append(semejanza)
        implicancias.append(implicancia)
    implicancias = np.array(implicancias)
    ind_lista = np.where(implicancias==np.max(implicancias))[0][0]
    indice, semejanza, implicancia = indices[ind_lista], semejanzas[ind_lista], implicancias[ind_lista]
    print(f'Implicancia: {implicancia:.3}, Semejanza: {semejanza:.3}')
    if implicancia >= umbral_implicancia:
        if semejanza>=umbral_semejanza:
            estrategia = 'pregunta_encontrada'
        else:
            estrategia = 'tema_encontrado'
    else:
        estrategia = 'pregunta_nueva'
    
    return estrategia, indice

def responde_pregunta(pregunta_usuario):
    imprime_tiempo('Definidas las funciones antes de empezar a preguntar')

    # Actualizar los dataframes de Respuestas y Fallos antes de usarlos
    query_respuestas = Respuestas.select()
    df_respuestas = pd.DataFrame(list(query_respuestas.dicts()))
    if len(df_respuestas) == 0:
        max_actual_id_respuestas = 0
    else:
        max_actual_id_respuestas = max(df_respuestas['id'])

    query_fallos = Fallos.select()
    df_fallos = pd.DataFrame(list(query_fallos.dicts()))
    if len(df_fallos) == 0:
        max_actual_id_fallos = 0
    else:
        max_actual_id_fallos = max(df_fallos['id'])

    preguntar_a_chatgpt, exito=False, False
    id_archivo_elegido = -1
    pagina_inicial, pagina_final = -1, -1
    estrategia,ind_df = determina_estrategia(pregunta_usuario,df_respuestas)
    imprime_tiempo('Determinada la estrategia para responder la pregunta')
    if estrategia=='pregunta_encontrada':
        print('Esta pregunta fue encontrada en las ya realizadas y por tanto usamos la respuesta existente')
        row = df_respuestas.loc[ind_df]
        id_pregunta, respuesta_final = row['id'], row['respuesta']
        exito=True
        respuesta = respuesta_final
        imprime_tiempo('Determinada la respuesta')
        query = Respuestas.update(frecuencia=Respuestas.frecuencia + 1).where(Respuestas.id == id_pregunta)
        query.execute()
        imprime_tiempo('Actualizada la tabla respuestas')

    elif estrategia=='tema_encontrado':
        print('Esta pregunta es similar a una existente y por tanto usamos las mismas paginas de input')
        row = df_respuestas.loc[ind_df]
        id_archivo_elegido, pagina_inicial, pagina_final = row['id'], row['pagina_inicial'], row['pagina_final']
        archivo_elegido=df_archivos[df_archivos["id"]==int(id_archivo_elegido)]["nombre"].values[0]
        imprime_tiempo('Determinado el archivo y las paginas a usar para responder')
        preguntar_a_chatgpt=True
    elif estrategia=='pregunta_nueva':
        print('Esta es una nueva pregunta')
        prompt_elige_archivo = hace_prompt_elige_archivo(pregunta_usuario, df_archivos)
        respuesta_elige_archivo = hacer_pregunta(prompt_elige_archivo)
        exito_archivo_elegido,archivo_elegido = determina_nombre_archivo_elegido(respuesta_elige_archivo, df_archivos)
        imprime_tiempo('Determinado el archivo a usar para responder')
        if exito_archivo_elegido:
            id_archivo_elegido = int(respuesta_elige_archivo)
            df_paginas_archivo = df_paginas[df_paginas['archivo']==id_archivo_elegido]
            n_paginas = int(df_archivos[df_archivos['nombre']==archivo_elegido]['paginas'].values[0])
            prompt_elige_paginas = hace_prompt_elige_paginas(pregunta_usuario, df_archivos, df_paginas_archivo)
            respuesta_elige_paginas = hacer_pregunta(prompt_elige_paginas)
            (exito_paginas_elegidas, preguntar_a_chatgpt,
            pagina_inicial, pagina_final) = determina_paginas_elegidas(respuesta_elige_paginas,n_paginas)
            imprime_tiempo('Determinadas las paginas a usar para responder')

    if preguntar_a_chatgpt is True:
        imprime_tiempo('Listos para hacer pregunta final a Chat GPT')
        prompt_final = hace_prompt_final(pregunta_usuario,archivo_elegido,pagina_inicial,pagina_final)
        respuesta_final = hacer_pregunta(prompt_final)
        imprime_tiempo('Hecha la pregunta final a Chat GPT')
        exito=True
        respuesta = respuesta_final
        nuevo_registro_respuestas = {
        'id': max_actual_id_respuestas+1,
        'pregunta': pregunta_usuario,
        'archivo': id_archivo_elegido,
        'pagina_inicial': pagina_inicial,
        'pagina_final': pagina_final,
        'frecuencia': 1,
        'nota_humano': -1,
        'respuesta': respuesta_final
        }
        # Agregar la nueva fila a la tabla
        Respuestas.create(**nuevo_registro_respuestas)
        imprime_tiempo('Actualizada la tabla respuestas')
    
    # Esto significa que es una pregunta nueva para la que no se pudo determinar
    # las páginas a usar como input
    if exito is False:
        respuesta = 'Lo siento, no logramos encontrar la respuesta a tu pregunta'
        estrategia, ind_df= determina_estrategia(pregunta_usuario,df_fallos)
        imprime_tiempo('Determinado si la pregunta fallida es nueva o no')
        #Esto significa que la pregunta fallida es nueva
        if estrategia!='pregunta_encontrada':
            max_actual_id_fallos += 1
            nuevo_registro_fallos = {
                'id': max_actual_id_fallos,
                'pregunta': pregunta_usuario,
                'archivo': id_archivo_elegido,
                'pagina_inicial': pagina_inicial,
                'pagina_final': pagina_final,
                'frecuencia': 1
            }
            Fallos.create(**nuevo_registro_fallos)
        else:
            row_fallos = df_fallos.loc[ind_df]
            id_pregunta_fallida = row_fallos['id']
            query = Fallos.update(frecuencia=Fallos.frecuencia + 1).where(Fallos.id == id_pregunta_fallida)
            query.execute()
        imprime_tiempo('Actualizada la tabla fallos')


    imprime_tiempo('Fin del procesamiento')
    return exito, respuesta



pregunta_usuario = "¿Cómo se crea una solicitud de servicio en caso de extravío del equipo?"
print(' ')
print(f'Pregunta: {pregunta_usuario}')

exito, respuesta = responde_pregunta(pregunta_usuario)
print(f'Respuesta: {respuesta}')