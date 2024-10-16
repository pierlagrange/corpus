FROM python:3.10-slim

RUN pip install --upgrade pip

COPY ./src /app/src/
COPY ./.env /app
COPY ./pyproject.toml /app
COPY ./src/corpus/ref_question /app/ref_question


# Offline gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=8000
ENV HF_HUB_OFFLINE=1
ENV DIFFUSERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV GRADIO_ANALYTICS_ENABLED=0
ENV DOCKER_MODE=1

WORKDIR /app
RUN pip install -e .

EXPOSE 8000

CMD ["python","src/corpus/setup_interface.py"]
