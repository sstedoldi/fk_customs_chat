from openai import OpenAI
import boto3
from jinja2 import Template
import json
import os

import logging

logger = logging.getLogger(__name__)

# Prompt Template
template_str = """
Respecto a la siguiente consulta: "{{ query }}"

**1. Evaluación Inicial:**
   - Determina si la consulta es una pregunta general o irrelevante para temas aduaneros.
   - Si la consulta incluye insultos o lenguaje ofensivo, responde educadamente indicando que no puedes procesarla.
   - Si la pregunta no está relacionada con normativa aduanera, responde indicando que solo puedes proporcionar asistencia en dicho ámbito.
   - Si la pregunta es ambigua o no está claramente formulada, solicita una reformulación para poder asistir mejor.

**2. Procesamiento de Documentos Relevantes:**
   - Si la consulta se refiere a normativa aduanera, revisa los documentos obtenidos del modelo de recuperación.
   - Cada documento tiene una puntuación de relevancia basada en su coincidencia con la consulta.

{% if documents %}
   - A continuación, se presentan los documentos más relevantes:
{% for doc, score in documents %}
     **Documento {{ loop.index }}:**  
     - **Título:** {{ doc.metadata["doc_title"] }}  
     - **Fuente:** {{ doc.metadata["source_path"] }}  
     - **Sección:** {{ doc.metadata["section"] }}  
     - **Capítulo:** {{ doc.metadata["chapter"] }}  
     - **Artículo:** {{ doc.metadata["article"] }}  
     - **Contenido relevante:**  
       "{{ doc.text | truncate(300) }}"  
     - **Puntuación de relevancia:** {{ 100 * score }}%  

{% endfor %}
{% else %}
   - No se encontraron documentos relevantes para esta consulta.
{% endif %}

**3. Generación de Respuesta:**
   - Redacta la respuesta basándote exclusivamente en los documentos relevantes.
   - Si los documentos relevantes tienen una puntuación inferior al **50%**, advierte al usuario sobre la posible falta de precisión en la respuesta.
   - No incluyas información ajena a la normativa aduanera.
   - Evita repetir conceptos innecesariamente y mantén una respuesta clara y objetiva.

"""

def simple_chat_openai(query, documents):
    """
    Calls OpenAI client to generate a response based on retrieved documents.
    """
    logger.info(f"Calling OpenAI LLM with query: {query}")
    logger.info(f"Documents retrieved: {documents}")

    # OpenAI API call
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sos un experto en leyes y normativa aduanera de Argentina. "
                        "Proporciona respuestas precisas y concisas basadas en la información relevante disponible."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more deterministic responses
            max_tokens=500,  # Prevent overly long responses
            top_p=0.9,  # Standard value for probabilistic sampling
            frequency_penalty=0.0,  # No penalty for repeating words
            presence_penalty=0.0  # No bias towards new topics
        )

        # Ensure a valid response exists before accessing message content
        if response.choices and len(response.choices) > 0:
            logger.info(f"Response from OpenAI: {response}")
            return response.choices[0].message.content.strip()
        else:
            return "No se pudo generar una respuesta adecuada."
    except Exception as e:
        logger.error(f"Error while calling OpenAI API: {e}")
        return "Ocurrió un error al generar la respuesta. Por favor, intenta nuevamente."


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