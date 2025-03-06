# Dockerfile
FROM python:3.11

# Set the working directory in the container
WORKDIR /usr/src/app

# # Install build-essential tools and gcc
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     python3-dev

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install farm-haystack[inference] and upgrade huggingface_hub
RUN pip install farm-haystack[inference]
RUN pip install --upgrade huggingface_hub
RUN python -m spacy download es_core_news_sm 

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run app.py when the container launches
CMD ["python", "app.py"]