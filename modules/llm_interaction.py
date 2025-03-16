from openai import OpenAI
import boto3
from jinja2 import Template
import json
import os

# Prompt Template
template_str = """
A continuación se presentan los documentos relevantes obtenidos por el modelo de recuperación, con sus respectivas puntuaciones de relevancia. 
Como experto en materia aduanera de Argentina, por favor proporciona una respuesta estrictamente basada en los documentos proporcionados.

{% for doc, score in documents %}
Documento {{ loop.index }}:
- Titulo: {{ doc.metatada_.doc_title }}
- Fuente: {{ doc.metatada_.source_path }}
- Información adicional: {{ doc.metatada_.additional_info }}
- Comentarios: {{ doc.metatada_.comments }}
- Contenido: {{ doc.content }}
- Puntuación de Relevancia: {{ 100*score }}

{% endfor %}

Responde la siguiente consulta, dando referencia sobre los documentos relevantes.
Si los documentos relevantes tienen un puntaje inferior al 70 %, advertí que es posible que la información no responda certeramente la pregunta. 

Evita comentarios subjetivos fuera de contexto y no repitas conceptos.

Consulta: {{ query }}
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

    # Render the prompt
    template = Template(template_str)
    prompt = template.render(query=query, documents=documents)

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
        
        return response_body.get("completion", "No response generated.")

    except Exception as e:
        return f"Error in AWS Bedrock call: {str(e)}"


def simple_chat_openai(query, documents):
    # OpenAI API call
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    # Render the prompt
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
