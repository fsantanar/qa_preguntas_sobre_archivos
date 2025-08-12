# QA sobre Presentaciones con LLM — Demo

Este repositorio contiene los **scripts principales** de un sistema de **preguntas y respuestas** sobre documentos en formato presentación (PPTX).  
Forma parte de un prototipo funcional desarrollado para un proyecto interno, y se publica para acompañar un [video de demostración](https://youtu.be/NBnySvRc61U) y como referencia en mi CV.

⚠️ **Nota**: Este repositorio no incluye datos reales ni credenciales. Está pensado para mostrar la arquitectura y el código principal, no para ser ejecutado directamente.

---

## Flujo general

El sistema sigue un pipeline de 6 pasos, más una aplicación de consulta:

### 1. `paso01_extraer_texto_de_ppts.py`
- Lee las presentaciones originales (.pptx).
- Por cada slide genera:
  - **Descripción técnica** (`PPTs_elementos/{archivo}/slideNNN_content.txt`): lista de textos e imágenes con su posición relativa.
  - **Imágenes de slide** (`PPTs_elementos/{archivo}/slideNNN_imageMM.png`).
  - **Slide original renderizada** (`PPTs_separados/{archivo}/slideN.png`).
- Observaciones: se guardan imágenes repetidas que se procesan en el siguiente paso.

### 2. `paso02_borra_imagenes_repetidas.py`
- Elimina imágenes duplicadas para ahorrar espacio.
- Genera un `imagenes_borradas.json` con el mapeo entre las borradas y sus originales.

### 3. `paso03_crea_csvs_contenido.py`
- Genera los insumos para la base de datos:
  - **Resumen preliminar del archivo** (LLM).
  - **Descripción completa de cada slide** (LLM + imagen opcional).
  - **Resumen breve de cada slide** (LLM).
- Guarda:
  - `Informacion_Archivos/Informacion_Paginas/{archivo}/Informacion_SlideNNN.txt`
  - `Informacion_Archivos/paginas.json` (todas las slides).
  - `Informacion_Archivos/archivos.json` (todos los archivos).
- Coste aproximado: ~400k tokens por archivo de 80 slides.

### 4. `paso04_creates_database.py`
- Crea la base de datos para almacenar:
  - Archivos
  - Páginas
  - Respuestas
  - Fallos

### 5. `paso05_creates_tables.py`
- Define las tablas y sus columnas.
- Esta estructura es usada para:
  - Determinar estrategia de respuesta (pregunta nueva / tema encontrado / pregunta encontrada).
  - Guardar contexto (archivo + páginas) y respuestas generadas.

### 6. `paso06_pregunta_chatgpt.py`
- Contiene funciones para interactuar con el modelo LLM.
- Incluye prompts para:
  - Elegir archivo relevante.
  - Elegir páginas relevantes.
  - Generar respuesta final.

### `chat_app_v05.py`
- Aplicación que recibe preguntas y entrega respuestas según el flujo:
  1. **Determinar estrategia**: reutilizar respuesta existente, reutilizar contexto, o buscar nuevo contexto.
  2. **Determinar contexto**: seleccionar archivo y páginas relevantes.
  3. **Generar respuesta**: a partir del contexto y la pregunta.
- Si falla en alguna etapa, se registra en la tabla de `fallos`.

---

## Configuración

`config.yml` incluye parámetros clave como:
- `decide_estrategia.umbral_implicancia` y `umbral_semejanza` para definir umbrales de decisión.
- `path_archivos` y `file_mode` para rutas y modo de lectura.
- `maximo_paginas` y `velocidad_maxima` para límites de procesamiento.

---

## Notas
- Este repo muestra únicamente los scripts y la configuración principal.
- No incluye:
  - Datos de entrada reales.
  - Claves de API.
  - Dependencias exactas.
- El objetivo es **dar trazabilidad** a la demo y evidenciar la arquitectura y lógica del sistema.

---

## Video de demostración
[Ver en YouTube](https://youtu.be/NBnySvRc61U)

