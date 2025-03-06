from openai import OpenAI
from jinja2 import Template
import os
    
def call_adubot_prueba1(query, documents):
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    template_str = """
    A continuación se presentan los documentos relevantes obtenidos por el modelo de recuperación, con sus respectivas puntuaciones de relevancia. 
    Como experto en materia aduanera de Argentina, por favor proporciona una respuesta estrictamente basada en los documentos proporcionados.

    {% for doc, score in documents %}
    Documento {{ loop.index }}:
    - Sección: {{ doc.meta.section }}
    - Capítulo: {{ doc.meta.chapter }}
    - Artículo: {{ doc.meta.articule }}
    - Contenido: {{ doc.content }}
    - Puntuación de Relevancia: {{ 100*score }}

    {% endfor %}

    Responde la siguiente consulta, dando referencia sobre los articulos evaluados (sin mendionar documentos fuentes ni puntajes de relevancia).
    Si los documentos relevantes tienen un puntaje inferior al 60 %, advertí que es  posible que la información no responda certeramente la pregunta. 
    
    Evita comentarios subjetivos fuera de contexto y no repitas conceptos.

    Consulta: {{ query }}
    """
    
    template = Template(template_str)
    prompt = template.render(query=query, documents=documents)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sos un experto en leyes y normativa aduanera de Argentina."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
