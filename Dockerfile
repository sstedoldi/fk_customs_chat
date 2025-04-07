# Dockerfile
FROM python:3.11-slim-bullseye

WORKDIR /fk_customs_chat

COPY requirements.txt /fk_customs_chat/

RUN python -m pip install --upgrade pip 

RUN pip install --no-cache-dir -r requirements.txt

COPY . /fk_customs_chat/

EXPOSE 8080

CMD ["python", "app.py"]
