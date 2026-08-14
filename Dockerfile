FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY src ./src

RUN python -m pip install .

EXPOSE 8000

CMD ["fastapi", "run", "--port", "8000"]