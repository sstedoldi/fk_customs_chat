from openai import OpenAI
import boto3
from jinja2 import Template
import json
import os

import logging

logger = logging.getLogger(__name__)

# Prompt Template
template_str = """
Respecto a la siguiente consulta: {{ query }}

1. Analiza si la misma se requiere a una pregunta general, filtra insultos y evita responder questiones agenas a asustos de normativa aduanera. Si la pregunta es inocente, responde advirtiendo que sos una herramienta profesional, que está ahí para asistir en temas de normativa aduanera.

2. Si la consulta se refiere a temas aduaneros, para responder, ten encuenta los documentos relevantes obtenidos por el modelo de recuperación, con sus respectivas puntuaciones de relevancia. 
Como experto en materia aduanera de Argentina, por favor proporciona una respuesta basada en los documentos proporcionados.

{% for doc, score in documents %}
Documento {{ loop.index }}:
- Título: {{ doc.metadata["doc_title"] }}
- Fuente: {{ doc.metadata["source_path"] }}
- Información adicional: {{ doc.metadata["additional_info"] }}
- Comentarios: {{ doc.metadata["comments"] }}
- Contenido: {{ doc.text }}
- Puntuación de Relevancia: {{ 100 * score }}%

{% endfor %}

Si los documentos relevantes tienen un puntaje inferior al 60 %, advertí que es posible que la información no responda certeramente la pregunta. 
Evita comentarios subjetivos fuera de contexto y no repitas conceptos.
"""

# AWS Bedrock setup
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

def simple_chat_aws(query, documents):
    """
    Calls AWS Bedrock's Meta Llama model to generate a response based on retrieved documents.
    """
    logger.info(f"Calling AWS LLM with query: {query}")
    logger.info(f"Documents retrieved: {documents}")

    # Convert NodeWithScore objects to (node, score) tuples
    docs_as_tuples = [(nws.node, nws.score) for nws in documents]

    # Render the prompt
    template = Template(template_str)
    prompt = template.render(query=query, documents=docs_as_tuples)

    # AWS Bedrock request payload
    payload = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.2,
        "top_p": 0.9
    }

    try:
        response = bedrock_runtime.invoke_model(
            body=json.dumps(payload),
            modelId="meta.llama3-8b-instruct-v1"
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        
        logger.info(f"Response from AWS: {response_body}")
        
        return response_body.get("completion", "No response generated.")

    except Exception as e:
        return f"Error in AWS Bedrock call: {str(e)}"


def simple_chat_openai(query, documents):
    logger.info(f"Calling OpenAI LLM with query: {query}")
    logger.info(f"Documents retrieved: {documents}")

    # OpenAI API call
    # client = OpenAI(api_key=os.environ['OPENAI_API_KEY']) 
    # ERROR:__main__:Error in /answer endpoint: Client.__init__() got an unexpected keyword argument 'proxies'
    client = OpenAI()
    
    # Convert NodeWithScore objects to (node, score) tuples
    docs_as_tuples = [(nws.node, nws.score) for nws in documents]

    # Render the prompt
    template = Template(template_str)
    prompt = template.render(query=query, documents=docs_as_tuples)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sos un experto en leyes y normativa aduanera de Argentina, listo para asistir a personas que quieran resolver dudas sobre temas aduaneros."},
            {"role": "user", "content": prompt}
        ]
    )

    logger.info(f"Response from OpenAI: {response}")

    return response.choices[0].message.content if response.choices else "No response generated"
