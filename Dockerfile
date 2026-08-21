FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY .env ./submission.env
COPY app/ ./app/

RUN python -m compileall -q app

USER 65534:65534

EXPOSE 8000

CMD ["python", "-m", "app.main"]
