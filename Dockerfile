# API image for Render. Deliberately small: xgboost is ~9 MB and the six
# boosters total 3.7 MB, so there is no need to re-implement tree inference to
# get a deployable service.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first so a code change does not reinstall the world.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The serving path and the artifacts it reads. Training scripts, tests, the
# frontend and the CLIP embeddings are excluded by .dockerignore.
COPY risk.py recommend.py shrinkage.py explain.py serve.py ./
COPY sundai_cgm/ ./sundai_cgm/
COPY artifacts/ ./artifacts/

ENV PYTHONUNBUFFERED=1

# Render supplies $PORT. Default to 8000 so `docker run -p 8000:8000` works too.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
