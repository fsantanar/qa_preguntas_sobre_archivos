# QA sobre Presentaciones con LLM — Demo

Este repositorio contiene los **scripts principales** de un sistema de **preguntas y respuestas** sobre documentos en formato presentación (PPTX).  
Forma parte de un prototipo funcional desarrollado para un proyecto interno, y se publica para acompañar un [video de demostración](https://youtu.be/NBnySvRc61U) y como referencia en mi CV.

⚠️ **Nota**: Este repositorio no incluye datos reales ni credenciales. Está pensado para mostrar la arquitectura y el código principal, no para ser ejecutado directamente sin datos y configuración.

---

## Flujo general y scripts

El sistema sigue un pipeline de 6 pasos, más dos modos de aplicación para hacer consultas (local y vía navegador):

### 1. `paso01_extraer_texto_de_ppts.py`
- Lee las presentaciones originales (.pptx).
- **Extrae y describe** cada slide:
  - Genera una **descripción técnica** (`PPTs_elementos/{archivo}/slideNNN_content.txt`) listando textos e imágenes con su posición relativa.
  - Guarda **todas las imágenes detectadas** por slide (`PPTs_elementos/{archivo}/slideNNN_imageMM.png`).
  - Guarda la **imagen renderizada** de cada slide (`PPTs_separados/{archivo}/slideN.png`).
- Observaciones: también captura imágenes repetidas que se filtran en el siguiente paso.

### 2. `paso02_borra_imagenes_repetidas.py`
- Detecta imágenes duplicadas mediante hash.
- Borra duplicados y genera `imagenes_borradas.json` con el mapeo de imágenes eliminadas y su original.
- Optimiza el espacio de almacenamiento sin alterar la información útil.

### 3. `paso03_crea_csvs_contenido.py`
- **Procesa y enriquece** la información extraída para construir la base de conocimiento:
  - Genera un **resumen preliminar del archivo** usando un modelo LLM.
  - Para cada slide, crea una **descripción completa** (integrando contexto previo, descripción técnica y, opcionalmente, la imagen).
  - Produce un **resumen breve** por slide, usado para búsquedas rápidas.
- Salidas:
  - `Informacion_Archivos/Informacion_Paginas/{archivo}/Informacion_SlideNNN.txt`
  - `Informacion_Archivos/paginas.json` (resúmenes de todas las slides).
  - `Informacion_Archivos/archivos.json` (resúmenes de todos los archivos).
- Controla el coste de tokens y tiempos de ejecución para lotes grandes.

### 4. `paso04_creates_database.py`
- Crea la base de datos definida en `config.yml`.
- No contiene credenciales hardcodeadas: el usuario/clave se leen de configuración o variables de entorno.

### 5. `paso05_creates_tables.py`
- Define las tablas:
  - `archivos` y `paginas`: metadatos y resúmenes.
  - `respuestas` y `fallos`: registro de interacciones exitosas y fallidas.
- Estructura usada para:
  - Determinar estrategia de respuesta (pregunta nueva, tema encontrado, pregunta encontrada).
  - Guardar contexto (archivo y páginas) y resultados.

### 6. `paso06_pregunta_chatgpt.py`
- Módulo para **hacer preguntas y obtener respuestas localmente** usando la base de conocimiento creada.
- Implementa prompts para:
  - Elegir archivo relevante.
  - Seleccionar páginas relevantes.
  - Generar la respuesta final con un modelo LLM.

### `chat_app_v05.py`
- Aplicación que expone el sistema vía navegador (interfaz web básica).
- Flujo:
  1. **Determinar estrategia**: reutilizar respuesta, reutilizar contexto o buscar contexto nuevo.
  2. **Determinar contexto**: seleccionar archivo y páginas relevantes.
  3. **Generar respuesta**: en base al contexto y la pregunta.
- Registra fallos cuando no es posible completar el flujo.
- Punto de partida para integrar el sistema en una página web.

---

## Configuración

- Todos los parámetros editables están en `config.yml`:
  - Límites de tokens y velocidad.
  - Umbrales de decisión para estrategias.
  - Unidades de medida para extracción de PPT.
  - Rutas de archivos.
- No hay claves API ni credenciales sensibles en el código.
- La clave del LLM (por ejemplo, OpenAI) se lee desde variables de entorno locales.

---

## Estructura del repositorio

```plaintext
qa_presentaciones_llm_demo/
├── paso01_extraer_texto_de_ppts.py
├── paso02_borra_imagenes_repetidas.py
├── paso03_crea_csvs_contenido.py
├── paso04_creates_database.py
├── paso05_creates_tables.py
├── paso06_pregunta_chatgpt.py
├── chat_app_v05.py
├── config.yml
├── docs/
│ └── Esquema_base_y_consultas_app.docx
└── README.md

```



---

## Video de demostración
[Ver en YouTube](https://youtu.be/NBnySvRc61U)

