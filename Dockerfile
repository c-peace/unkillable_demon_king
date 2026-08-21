FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

# The evaluation harness starts the container with `docker run -p 8000:8000 <image>`
# and injects no environment, so the team key has to travel with the image or every
# chat completion fails with 503 service_not_configured.
# A runtime `-e LUNIT_FM_API_KEY=...` still overrides this value.
ENV LUNIT_FM_API_KEY=lunit_qfF2amixPkf2fGoRu1RdTxTzPQL-jpnwB9YuEIbaBEk

WORKDIR /app

COPY app/ ./app/

RUN python -m compileall -q app

USER 65534:65534

EXPOSE 8000

CMD ["python", "-m", "app.main"]
