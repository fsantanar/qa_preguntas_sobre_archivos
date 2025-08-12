
import time
ti=time.time()
def imprime_tiempo(mensaje):
    tf = time.time()
    dt = tf-ti
    print(f'Estado: {mensaje}, Time Covered: {dt:.2f}')
    return tf
t_inicio = imprime_tiempo('Inicio')
import openai
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from peewee import Model, IntegerField, TextField, CharField, PostgresqlDatabase, FloatField, DateTimeField, BooleanField
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging
logging.set_verbosity_error()
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from datetime import datetime
import warnings
from reportlab.pdfgen import canvas
from PIL import Image 
from flask import Flask, request, Response, render_template, stream_with_context, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
from reportlab.pdfgen import canvas


t_modulos = imprime_tiempo('Cargados los Módulos')

app = Flask(__name__)
CORS(app)



# Suprimir el FutureWarning sobre clean_up_tokenization_spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")



### IMPORTANTE:
#
# Descubrir si motivo de mayor respuesta final es por prompt, ver cuanto demora el archivo,o si es otro motivo,
# Comparar nuestra efectividad con la de la web de chat gpt repitiendo prompt y ver que tanto cambia usando imagenes en vez de descripción para ver si
# vale la pena cambiar precio por efectividad (y a lo mejor tiempo)
# Analizar todo esquema de entrenamiento y prompt para tener resumen y ver donde se pede mejorar tiempo, ahorro en tokens y efectividad. 

# Cargar la configuración desde el archivo YAML
with open(f'config.yml', 'r') as file:
    config = yaml.safe_load(file)



# Obtener el path del archivo actual
code_path = os.path.abspath(__file__)
# Obtener el nombre de la base de config.yml
db_name = config['database']['name']
# Obtener el nombre del archivo
nombre_codigo = os.path.basename(code_path)



class Config:
    path_archivos = None
    umbral_semejanza = None
    umbral_implicancia = None
    maximo_paginas = None

    @classmethod
    def cargar_configuracion(cls, archivo_config):
        with open(archivo_config, 'r') as file:
            config = yaml.safe_load(file)
        cls.umbral_semejanza = config['decide_estrategia']['umbral_semejanza']
        cls.umbral_implicancia = config['decide_estrategia']['umbral_implicancia']
        cls.maximo_paginas = config['maximo_paginas']
        cls.path_archivos = config['path_archivos']

# Cargar configuración de valores constantes para toda la ejecución
Config.cargar_configuracion('config.yml')



# Configurar la conexión a la base de datos PostgreSQL
db = PostgresqlDatabase(db_name, user='felipesantana', password='', host='localhost')

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
    id = IntegerField(primary_key=True)
    archivo = IntegerField()
    n_pagina = IntegerField()
    descripcion = TextField()

    class Meta:
        table_name = 'paginas'


class Respuestas(BaseModel):
    id = IntegerField(primary_key=True)
    pregunta = TextField()
    archivo = IntegerField()
    pagina_inicial = IntegerField()
    pagina_final = IntegerField()
    id_pregunta_mismo_tema = IntegerField()
    frecuencia = IntegerField()
    tokens_estrategia = IntegerField()
    tokens_contexto_archivo = IntegerField()
    tokens_contexto_paginas = IntegerField()
    tokens_respuesta = IntegerField()
    segundos_estrategia_total = IntegerField()
    segundos_contexto_archivo = IntegerField()
    segundos_contexto_paginas = IntegerField()
    segundos_respuesta = IntegerField()
    segundos_estrategia_ultima = IntegerField()
    comparaciones_estrategia_ultima = IntegerField()
    momento_primera_respuesta = DateTimeField()
    momento_ultima_respuesta = DateTimeField()
    version_codigo = CharField(max_length=20)
    nota_humano = FloatField()
    respuesta = TextField()
    class Meta:
        table_name = 'respuestas'



class Fallos(BaseModel):
    id = IntegerField(primary_key=True)
    pregunta = TextField()
    momento_respuesta = DateTimeField()
    version_codigo = CharField(max_length=20)
    exito_contexto_archivo = BooleanField()
    exito_contexto_paginas = BooleanField()
    archivo = TextField()
    pagina_inicial = IntegerField()
    pagina_final = IntegerField()
    tokens_estrategia = IntegerField()
    tokens_contexto_archivo = IntegerField()
    tokens_contexto_paginas = IntegerField()
    tokens_respuesta = IntegerField()
    segundos_estrategia = IntegerField()
    segundos_contexto_archivo = IntegerField()
    segundos_contexto_paginas = IntegerField()
    segundos_respuesta = IntegerField()

    class Meta:
        table_name = 'fallos'


# Cargar los dataframes inmutables para agrgarlos despues como atributos de clase
dataframe_archivos = pd.DataFrame(list(Archivos.select().dicts()))
dataframe_paginas = pd.DataFrame(list(Paginas.select().dicts()))


# Reemplaza con tu API key de OpenAI
api_key =os.environ['OPENAI_API_KEY']

# Configura la API key
openai.api_key = api_key
t_tablas = imprime_tiempo('Cargadas las tablas, Antes de cargar modelos')
# Modelo de semejanza de frases
modelo_semejanza = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
t_primer_modelo = imprime_tiempo('Definido el modelo de Semejanza')
# Modelo de implicancia semántica
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
modelo_implicancia = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
t_ambos_modelos = imprime_tiempo('Definido el modelo de Implicancia')


def generar_pdf_imagenes(carpeta, imagenes, nombre_pdf):
    pdf_path = f'static/{nombre_pdf}'
    c = None  # Inicializamos el canvas en None para poder ajustarlo según las dimensiones de la primera imagen

    for imagen in imagenes:
        imagen_path = os.path.join(f'static/imagenes/{carpeta}', imagen)
        
        # Abrimos la imagen para obtener sus dimensiones originales
        with Image.open(imagen_path) as img:
            img_width, img_height = img.size

        # Si aún no se ha creado el canvas, lo creamos con el tamaño de la primera imagen
        if c is None:
            c = canvas.Canvas(pdf_path, pagesize=(img_width, img_height))

        # Ajustamos el tamaño de la página al de cada imagen
        c.setPageSize((img_width, img_height))

        # Dibujamos la imagen en el PDF ocupando todo el espacio disponible
        c.drawImage(imagen_path, 0, 0, width=img_width, height=img_height)
        c.showPage()  # Crear una nueva página para cada imagen

    if c:
        c.save()  # Guardar el PDF solo si el canvas fue creado

    return pdf_path



def imprimir_mensaje(texto, velocidad=0.3):
    palabras = texto.split()  # Dividir en palabras
    for palabra in palabras:
        print(palabra, end=' ', flush=True)
        time.sleep(velocidad)
    print()

def resalta_mensaje(mje):
    largo = len(mje)
    print(' ')
    print('#'*(largo+10))
    print('###  '+mje+'  ###')
    print('#'*(largo+10))
    print(' ')

def calcula_maxima_id(dataframe):
    if len(dataframe) == 0:
        max_id = 0
    else:
        max_id = max(dataframe['id'])
    return max_id

def son_semejantes(frase1, frase2):
    # Genera embeddings para las frases
    embedding1 = modelo_semejanza.encode(frase1, convert_to_tensor=True)
    embedding2 = modelo_semejanza.encode(frase2, convert_to_tensor=True)

    # Calcula la similitud coseno
    semejanza = util.cos_sim(embedding1, embedding2).item()

    # Retorna True si la similitud es mayor al umbral
    return semejanza

def son_implicantes(frase1, frase2):
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
    return entailment_prob_final, contradiction_prob_final


class Pregunta:
    df_archivos = None
    df_paginas = None


    def __init__(self, pregunta_usuario, df_respuestas, df_fallos):
        self.pregunta_usuario = pregunta_usuario
        self.df_respuestas = df_respuestas
        self.df_fallos = df_fallos
        self.max_actual_id_respuestas = calcula_maxima_id(self.df_respuestas)
        self.max_actual_id_fallos = calcula_maxima_id(self.df_fallos)


        self.prompt_estrategia = None
        self.resultado_estrategia = None
        self.output_estrategia = None
        self.estrategia = None
        self.semejanza = 0
        self.implicancia = 0
        self.id_respuesta = -1
        self.id_pregunta_mismo_tema = -1
        self.id_mas_parecida = -1
        self.n_comparaciones = 0
        self.tokens_estrategia = 0
        self.dt_estrategia = 0

        self.prompt_contexto_archivo = None
        self.resultado_contexto_archivo = None
        self.prompt_contexto_paginas = None
        self.resultado_contexto_paginas = None
        self.output_contexto = None
        self.archivo_usado = None
        self.id_archivo_usado = -1
        self.pagina_inicial = -1
        self.pagina_final = -1
        self.exito_contexto_archivo = False
        self.exito_contexto_paginas = False
        self.tokens_contexto_archivo = 0
        self.tokens_contexto_paginas = 0
        self.tokens_contexto = 0
        self.dt_contexto_archivo = 0
        self.dt_contexto_paginas = 0
        self.dt_contexto = 0

        self.prompt_respuesta = None
        self.resultado_respuesta = None
        self.output_respuesta = None
        self.tokens_respuesta = 0
        self.dt_respuesta = 0


    @classmethod
    def cargar_dataframes(cls, df_archivos, df_paginas):
        cls.df_archivos = df_archivos
        cls.df_paginas = df_paginas
            

    def hace_prompt_elige_archivo(self):
        # A partir de la pregunta original del usuario y el dataframe con la info de los archivos
        # Crea el texto de la pregunta para elegir el archivo necesario para responder la pregunta
        # del usuario
        prompt = (f'Necesito eventualmente responder la pregunta "{self.pregunta_usuario}" usando '
                  f'la información contenida en un set de archivos. A continuación te muestro la id, '
                  f'el nombre y un resumen de cada archivo. Quiero que me respondas solo con la id del '
                  f'archivo más útil para responder la pregunta (y si ninguno sirve que respondas '
                  f'"-1")\n')
        for __,row in Pregunta.df_archivos.iterrows():
            prompt += (f'id={row["id"]}, nombre={row["nombre"]}, resumen={row["resumen"]}\n')
        self.prompt_contexto_archivo = prompt


    def determina_nombre_archivo_a_usar(self):
        
        # Primero quitamos el espacio blanco
        respuesta_a_usar = self.resultado_contexto_archivo.replace(' ','')

        # Si la respuesta contiene 'id=' tomamos solo lo que viene despues de eso
        if "id=" in respuesta_a_usar:
            respuesta_a_usar = respuesta_a_usar.split("id=")[1]

        # Comprobar si la respuesta es un número y corresponde a una id valida de un archivo
        if not respuesta_a_usar.isdigit():
            print(f'La respuesta de ChatGPT ({self.resultado_contexto_archivo}) sobre el archivo '
                  f'no fue entregada como número')
        elif int(respuesta_a_usar) not in Pregunta.df_archivos['id'].values:
            print(f'El número de archivo entregado por Chat GPT ({self.resultado_contexto_archivo}) '
                  f'no corresponde a una ID válida')
        else:
            self.exito_contexto_archivo = True
            respuesta_a_usar = int(respuesta_a_usar)
            self.id_archivo_usado = respuesta_a_usar
            self.archivo_usado = Pregunta.df_archivos[Pregunta.df_archivos["id"] == respuesta_a_usar]["nombre"].values[0]
            print(f'El archivo necesario para responder la pregunta según Chat GPT es "{self.archivo_usado}" (ID={self.id_archivo_usado})')


    def hace_prompt_elige_paginas(self):
        # A partir de la pregunta original del usuario y el dataframe con las paginas del archivo
        # que eligio chat gpt como util para la respuesta, se crea el texto para pedirle que eliga
        # el rango de paginas necesario para responder la pregunta
        prompt = (f'Necesito identificar el rango de páginas del archivo "{self.archivo_usado}" '
                  f'necesario para responder la pregunta: "{self.pregunta_usuario}". Para eso a '
                  f'continuación te indico para cada página un resumen de su contenido.\n\n')
        
        df_paginas_archivo = Pregunta.df_paginas[Pregunta.df_paginas['archivo'] == self.id_archivo_usado]
        for __,row in df_paginas_archivo.iterrows():
            prompt += (f'Página {row["n_pagina"]}: "{row["descripcion"]}"\n')
        prompt += (f'\nBasado en esta información, responde exclusivamente con el rango de páginas que '
                f'son más útiles para responder la pregunta: "{self.pregunta_usuario}". Si no hay páginas útiles, '
                f'responde solo con "-1". La respuesta debe ser solo el rango de páginas, '
                f'por ejemplo, "1-1".')

        self.prompt_contexto_paginas = prompt


    def determina_paginas_elegidas(self):
        # Esta función determina que se hace con la respuesta de Chat GPT de las paginas elegidas
        # Si la respuesta es valida se intenta responder la pregunta con ese rango de paginas,
        # si no es valida se intenta usar el archivo completo.
        # Si el rango a intentar no es demasiado largo se hace la pregunta a Chat GPT y por tanto
        # respond es True, pero si el rango a intentar es demasiado largo no se hace la pregunta
        # para ahorrar tokens.
        found = False # Esto indica si se pudo obtener un rango de paginas a partir de la respuesta
        pagina_inicial, pagina_final = -1, -1
        n_paginas = len(Pregunta.df_paginas[Pregunta.df_paginas['archivo'] == self.id_archivo_usado])
        respuesta_sep = self.resultado_contexto_paginas.split('-')

        if len(respuesta_sep)!=2 or not respuesta_sep[0].isdigit() or not respuesta_sep[1].isdigit():
            print(f'La respuesta de paginas ({self.resultado_contexto_paginas}) no corresponde a un '
                  f' rango de números')
        elif ((int(respuesta_sep[0])<=0 or int(respuesta_sep[0])>n_paginas)
            or (int(respuesta_sep[1])<=0 or int(respuesta_sep[1])>n_paginas)):
            print(f'El rango de páginas entregado por ({self.resultado_contexto_paginas}) cae '
                  f'fuera del rango total del archivo')
        else:
            found = True
            pagina_inicial = int(respuesta_sep[0])
            pagina_final = int(respuesta_sep[1])
            print(f'El rango de páginas propuesto para responder la pregunta es '
                  f'de la {int(respuesta_sep[0])} a la {int(respuesta_sep[1])}')
            
        # Ahora determinamos si se va a hacer la pregunta final o no
        # dependiendo del largo en paginas que deberiamos usar de input
        if found is False: # Si no se encuentra el rango de paginas más util se intenta el archivo entero
            pagina_inicial, pagina_final=1, n_paginas
            print('Como Chat GPT no pudo encontrar el rango de páginas más utiles intentaremos '
                  'usar el archivo completo')

        n_paginas_input = 1 + pagina_final - pagina_inicial

        if n_paginas_input <= Config.maximo_paginas:
            self.exito_contexto_paginas=True
            self.pagina_inicial = pagina_inicial
            self.pagina_final = pagina_final
            print(f'Como el rango de páginas a usar es suficientemente corto lo usaremos como input')
        else:
            print(f'Como el rango de páginas a usar es demsiado largo no se puede responder la pregunta')


    def hace_prompt_final(self):
        # A partir de la pregunta original del usuario, el archivo y el tema seleccionados
        # por chat gpt como utiles para la respuesta se crea el texto para obtener la respuesta final

        contexto = (f'Necesito responder la pregunta "{self.pregunta_usuario}" usando la información '
                    f'contenida en un archivo llamado "{self.archivo_usado}" que trata de distintos temas. '
                    f'Dentro de él se determinó el rango de páginas más útiles para responder la pregunta. '
                    f'Por tanto abajo te pego la descripción detallada del contenido de cada página '
                    f'perteneciente a ese rango, para que las uses para responder la pregunta '
                    f'del usuario.\n\n')

        contenido_paginas=''
        for pagina in range(self.pagina_inicial,self.pagina_final+1):
            path_archivo = f'{Config.path_archivos}/{self.archivo_usado}/Informacion_Slide{pagina:03}.txt'
            archivo_actual = open(path_archivo,'r')
            contenido_pagina = archivo_actual.read()
            archivo_actual.close()
            contenido_paginas += contenido_pagina
        
        prompt = contexto + contenido_paginas
        self.prompt_respuesta = prompt


    def hacer_pregunta(self, tipo_pregunta, modelo="gpt-4o-mini"):
        """
        Realiza una consulta a la API de OpenAI según el tipo de 
        pregunta (estrategia, contexto_archivo, contexto_paginas, respuesta).
        
        Parámetros:
        - tipo_pregunta: 'estrategia', 'contexto_archivo', 'contexto_paginas', 'respuesta'.
        - modelo: Modelo de OpenAI a utilizar (por defecto: 'gpt-4o-mini').

        Asigna los prompts, resultados y tokens como atributos del objeto, según el tipo de pregunta.
        Si el tipo de pregunta es contexto los tokens 
        
        Retorna:
        - resultado: La respuesta generada por la API de OpenAI.
        - tokens: El número de tokens utilizados en la consulta.
        """        
        try:
            map_atributos = {
                'estrategia': {'prompt': 'prompt_estrategia',
                               'resultado': 'resultado_estrategia',
                               'tokens': 'tokens_estrategia'},
                'contexto_archivo': {'prompt': 'prompt_contexto_archivo',
                                     'resultado': 'resultado_contexto_archivo',
                                     'tokens': 'tokens_contexto_archivo'},
                'contexto_paginas': {'prompt': 'prompt_contexto_paginas',
                                     'resultado': 'resultado_contexto_paginas',
                                     'tokens': 'tokens_contexto_paginas'},
                'respuesta': {'prompt': 'prompt_respuesta',
                              'resultado': 'resultado_respuesta',
                              'tokens': 'tokens_respuesta'}
            }

            pregunta = getattr(self, map_atributos[tipo_pregunta]['prompt'])

            response = openai.chat.completions.create(
                model=modelo, 
                messages=[{"role": "user", "content": pregunta}]
            )
            resultado = response.choices[0].message.content
            tokens = response.usage.total_tokens

            setattr(self, map_atributos[tipo_pregunta]['resultado'], resultado)
            setattr(self, map_atributos[tipo_pregunta]['tokens'], tokens)

            return resultado, tokens

        except Exception as e:
            return f"Error al hacer la pregunta: {e}", 0

    def calcula_mensaje_estrategia(self):
        if self.estrategia=='pregunta_encontrada':
            output = ('Esta pregunta ha sido determinada como análoga de una pregunta frecuente y '
                      'por tanto utilizaremos esa respuesta')
        elif self.estrategia=='tema_encontrado':
            output = ('Esta pregunta ha sido determinada como similar a una pregunta frecuente, por '
                      'tanto usaremos el mismo contexto de páginas utilizado para esa pregunta')
        elif self.estrategia=='pregunta_nueva':
            output = ('Esta es una pregunta nueva y por tanto intentaremos responderla usando la base de información')
        self.output_estrategia = output

    def determina_estrategia(self, umbral_semejanza = Config.umbral_semejanza,
                             umbral_implicancia = Config.umbral_implicancia):
        # Determinar estrategia a partir de semejanza e implicancia
        ti_estrategia = time.time()
        estrategia = 'pregunta_nueva'
        tokens = 0 # Porque por ahora se hace sin Chat GPT pero puede que esto cambie
        ids_todas, implicancias_todas = [-1], [0]
        max_semejanza_candidatas, implicancia_candidata = 0, 0
        id_pregunta_mismo_tema = -1
        df_res = self.df_respuestas

        for __, row in df_res.iterrows():
            implicancia, __ = son_implicantes(self.pregunta_usuario, row['pregunta'])
            id_actual = row['id']
            implicancias_todas.append(implicancia)
            ids_todas.append(id_actual)
            # Solo hacemos algo si la implicancia es mayor que el limite
            if implicancia > umbral_implicancia:
                semejanza = son_semejantes(self.pregunta_usuario, row['pregunta'])
                if semejanza > umbral_semejanza:
                    estrategia = 'pregunta_encontrada'
                    self.estrategia = 'pregunta_encontrada'
                    self.id_respuesta = id_actual
                    self.implicancia = implicancia
                    self.semejanza = semejanza
                    break
                else:
                    if semejanza>max_semejanza_candidatas:
                        estrategia = 'tema_encontrado'
                        max_semejanza_candidatas = semejanza
                        implicancia_candidata = implicancia
                        id_pregunta_mismo_tema = id_actual
        
        self.n_comparaciones = len(implicancias_todas) - 1


        if estrategia == 'tema_encontrado':
            self.estrategia = 'tema_encontrado'
            self.id_pregunta_mismo_tema = id_pregunta_mismo_tema
            self.semejanza = max_semejanza_candidatas
            self.implicancia = implicancia_candidata
        
        elif estrategia == 'pregunta_nueva':
            self.estrategia = 'pregunta_nueva'
            mayor_implicancia = max(implicancias_todas)
            id_mayor_implicancia = ids_todas[np.where(np.array(implicancias_todas)==mayor_implicancia)[0][0]]
            if len(df_res)==0:
                semejanza_mayor_implicancia = 0
            else:
                pregunta_mayor_implicancia = df_res[df_res['id']==id_mayor_implicancia]['pregunta'].values[0]
                semejanza_mayor_implicancia = son_semejantes(self.pregunta_usuario, pregunta_mayor_implicancia)
            self.semejanza = semejanza_mayor_implicancia
            self.implicancia = mayor_implicancia
            self.id_mas_parecida = id_mayor_implicancia
        
        tf_estrategia = time.time()
        self.dt_estrategia = tf_estrategia - ti_estrategia
        self.tokens_estrategia = tokens # Por ahora solo asignando 0 pero para recordar si cambia
        self.calcula_mensaje_estrategia()


    def calcula_mensaje_contexto(self):
        if self.estrategia == 'pregunta_nueva' and self.exito_contexto_archivo is False:
            mensaje = f'Lo siento pero no pude encontrar el archivo necesario para responder la pregunta'

        elif self.estrategia == 'pregunta_nueva' and self.exito_contexto_paginas is False:
            mensaje = f'Lo siento pero no pude encontrar las páginas necesarias para responder la pregunta'

        else:
            if self.pagina_final == self.pagina_inicial:
                paginas = f'la página {self.pagina_inicial} '
            else:
                paginas = f'las páginas {self.pagina_inicial} a {self.pagina_final} '

            mensaje = f'Buscaremos la respuesta en {paginas} del archivo "{self.archivo_usado}"...'
        
        self.output_contexto = mensaje


    def determina_contexto(self):
        ti_contexto = time.time()
        df_res = self.df_respuestas
        df_arch = Pregunta.df_archivos

        # Primero vemos si la informacion del contexto la sacamos de preguntas antiguas
        # segun el valor de la estrategia
        busqueda_necesaria = True

        if self.estrategia == 'pregunta_encontrada':
            busqueda_necesaria = False
            row_resp = df_res[df_res['id'] == self.id_respuesta]
            
        elif self.estrategia == 'tema_encontrado':
            busqueda_necesaria = False
            row_resp = df_res[df_res['id'] == self.id_pregunta_mismo_tema]
        
        # Si por estrategia no hay que buscar asignamos directamente los valores
        if busqueda_necesaria is False:
            self.pagina_inicial = row_resp['pagina_inicial'].values[0]
            self.pagina_final = row_resp['pagina_final'].values[0]
            self.id_archivo_usado = row_resp['archivo'].values[0]
            self.archivo_usado = df_arch[df_arch["id"]==self.id_archivo_usado]["nombre"].values[0]

        # Si la estrategia es pregunta nueva efectivamente buscamos el contexto
        else:
            self.hace_prompt_elige_archivo()
            __, __ = self.hacer_pregunta('contexto_archivo')

            self.determina_nombre_archivo_a_usar()
            tf_contexto_archivo = time.time()
            self.dt_contexto_archivo = tf_contexto_archivo - ti_contexto
            # Si logro encontrar el archivo buscamos las paginas
            if self.exito_contexto_archivo is True:
                self.hace_prompt_elige_paginas()
                __, __ = self.hacer_pregunta('contexto_paginas')
                self.determina_paginas_elegidas()
                tf_contexto_paginas = time.time()
                self.dt_contexto_paginas = tf_contexto_paginas - tf_contexto_archivo
                
        self.dt_contexto = self.dt_contexto_archivo + self.dt_contexto_paginas
        self.tokens_contexto = self.tokens_contexto_archivo + self.tokens_contexto_paginas
        self.calcula_mensaje_contexto()


    def calcula_mensaje_respuesta(self):

        #Entregamos un tipo de mensaje si la respuesta es fallida y otro si es exitosa
        if self.estrategia == 'pregunta_nueva' and self.exito_contexto_paginas is False:
            self.output_respuesta = 'Respuesta Fallida'

        else:
            self.output_respuesta = self.resultado_respuesta
            df_arch = Pregunta.df_archivos
            npaginas_archivo = df_arch[df_arch['nombre']==self.archivo_usado]['paginas'].values[0]
            num_ceros_paginas = len(str(npaginas_archivo))
            imagenes = [f'slide-{i:0{num_ceros_paginas}}.png'
                        for i in range(self.pagina_inicial, self.pagina_final + 1)]
            nombre_pdf = f'{self.archivo_usado}_{self.pagina_inicial}_a_{self.pagina_final}.pdf'
            __ = generar_pdf_imagenes(self.archivo_usado, imagenes, nombre_pdf)

            enlace_pdf = f'/static/{nombre_pdf}'
            self.output_respuesta += (f'<br><br>Puedes ver/descargar el archivo PDF con las imágenes del rango de '
                                    f'páginas usadas de la fuente haciendo clic en el siguiente enlace: '
                                    f'<a href="{enlace_pdf}" target="_blank">link_PDF</a>')

    def determina_respuesta(self):
        ti_respuesta = time.time()

        if self.estrategia == 'pregunta_encontrada':
            df_res = self.df_respuestas
            row_respuestas = df_res[df_res['id']==self.id_respuesta]
            self.resultado_respuesta = row_respuestas['respuesta'].values[0]
            self.calcula_mensaje_respuesta()

            query = (Respuestas.update(frecuencia=Respuestas.frecuencia + 1,
                                       tokens_estrategia=Respuestas.tokens_estrategia + self.tokens_estrategia,
                                       segundos_estrategia_total = Respuestas.segundos_estrategia_total + round(self.dt_estrategia),
                                       segundos_estrategia_ultima = round(self.dt_estrategia),
                                       comparaciones_estrategia_ultima = self.n_comparaciones,
                                       momento_ultima_respuesta = datetime.now(),
                                       version_codigo = nombre_codigo)
                                       .where(Respuestas.id == self.id_respuesta))
            query.execute()

        # Si vamos a preguntar a la IA (si fue exitosa las paginas implica que lo fue el archivo)
        elif ((self.estrategia == 'tema_encontrado') or
              (self.estrategia == 'pregunta_nueva' and self.exito_contexto_paginas is True)):
            self.hace_prompt_final()
            resultado_respuesta, __ = self.hacer_pregunta('respuesta')
            self.resultado_respuesta = resultado_respuesta
            self.calcula_mensaje_respuesta()
            tf_respuesta = time.time()
            self.dt_respuesta = tf_respuesta - ti_respuesta

            nuevo_registro_respuestas = {
            'id': self.max_actual_id_respuestas+1,
            'pregunta': self.pregunta_usuario,
            'archivo': self.id_archivo_usado,
            'pagina_inicial': self.pagina_inicial,
            'pagina_final': self.pagina_final,
            'id_pregunta_mismo_tema': self.id_pregunta_mismo_tema ,
            'frecuencia': 1,
            'tokens_estrategia': 0,
            'tokens_contexto_archivo': self.tokens_contexto_archivo,
            'tokens_contexto_paginas': self.tokens_contexto_paginas,
            'tokens_respuesta': self.tokens_respuesta,
            'segundos_estrategia_total': round(self.dt_estrategia),
            'segundos_contexto_archivo': round(self.dt_contexto_archivo),
            'segundos_contexto_paginas': round(self.dt_contexto_paginas),
            'segundos_respuesta': round(self.dt_respuesta),
            'segundos_estrategia_ultima': round(self.dt_estrategia),
            'comparaciones_estrategia_ultima': self.n_comparaciones,
            'momento_primera_respuesta': datetime.now(),
            'momento_ultima_respuesta': datetime.now(),
            'version_codigo': nombre_codigo,
            'nota_humano': -1,
            'respuesta': self.resultado_respuesta
            }
            # Agregar la nueva fila a la tabla
            Respuestas.create(**nuevo_registro_respuestas)


        # Si intentamos buscar el contexto y fallamos
        elif self.estrategia == 'pregunta_nueva' and self.exito_contexto_paginas is False:
            self.calcula_mensaje_respuesta()
            self.max_actual_id_fallos += 1
            nuevo_registro_fallos = {
                'id': self.max_actual_id_fallos,
                'pregunta': self.pregunta_usuario,
                'momento_respuesta': datetime.now(),
                'version_codigo': nombre_codigo,
                'exito_contexto_archivo': self.exito_contexto_archivo,
                'exito_contexto_paginas': self.exito_contexto_paginas,
                'archivo': self.id_archivo_usado,
                'pagina_inicial': self.pagina_inicial,
                'pagina_final': self.pagina_final,
                'tokens_estrategia': self.tokens_estrategia,
                'tokens_contexto_archivo': self.tokens_contexto_archivo,
                'tokens_contexto_paginas': self.tokens_contexto_paginas,
                'tokens_respuesta': self.tokens_respuesta,
                'segundos_estrategia': self.dt_estrategia,
                'segundos_contexto_archivo': self.dt_contexto_archivo,
                'segundos_contexto_paginas': self.dt_contexto_paginas,
                'segundos_respuesta': self.dt_respuesta
            }
            Fallos.create(**nuevo_registro_fallos)




# Asignar a la clase Pregunta
Pregunta.cargar_dataframes(dataframe_archivos, dataframe_paginas)




@app.route('/')
def index():
    return render_template('chatbot.html')

# Ruta para servir archivos con subcarpetas
@app.route('/image/<folder>/<filename>')
def image(folder, filename):
    return send_from_directory(f'static/imagenes/{folder}', filename)

@app.route('/download_pdf/<nombre_pdf>', methods=['GET'])
def descargar_pdf(nombre_pdf):
    pdf_path = f'static/{nombre_pdf}'
    response = send_file(pdf_path, as_attachment=True)

    # Después de servir el archivo, lo eliminamos
    try:
        os.remove(pdf_path)
        print(f'{pdf_path} eliminado después de la descarga.')
    except Exception as e:
        print(f'Error al eliminar {pdf_path}: {e}')

    return response

# Streaming de respuestas por etapas
@app.route('/pregunta', methods=['POST'])
def handle_question():
    pregunta_usuario = request.json['pregunta']
    resalta_mensaje(f"Pregunta: {pregunta_usuario}")

    # Actualizar los dataframes de Respuestas y Fallos cad vez que se hace una pregunta antes de usarlos
    dataframe_respuestas = pd.DataFrame(list(Respuestas.select().dicts()))
    dataframe_fallos = pd.DataFrame(list(Fallos.select().dicts()))

    # Creamos el objeto de la clase Pregunta
    objeto_pregunta = Pregunta(pregunta_usuario, dataframe_respuestas, dataframe_fallos)

    def generar_respuestas():
        # 1. Determinar la estrategia
        imprime_tiempo('Comenzando a determinar estrategia')
        objeto_pregunta.determina_estrategia()
        respuesta_estrategia = objeto_pregunta.output_estrategia
        yield f'data: {respuesta_estrategia}<END>\n\n'

        # 2. Determinar archivo y páginas
        imprime_tiempo('Comenzando a determinar archivo y paginas a usar')
        objeto_pregunta.determina_contexto()
        respuesta_contexto = objeto_pregunta.output_contexto
        yield f'data: {respuesta_contexto}<END>\n\n'

        # 3. Generar la respuesta final
        imprime_tiempo('Comenzando a Responder Pregunta Final')
        objeto_pregunta.determina_respuesta()
        respuesta_final = objeto_pregunta.output_respuesta
        print(f'Respuesta final: {respuesta_final}')
        imprime_tiempo('Respondida la Pregunta')
        yield f'data: Respuesta final: {respuesta_final}<END>\n\n'

    return Response(stream_with_context(generar_respuestas()), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=False, port=8080)

