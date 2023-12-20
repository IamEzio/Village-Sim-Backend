FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

ENV APP_HOME /python/src/backend
ENV PORT 8080
ENV FLASK_RUN_PORT "$PORT"
WORKDIR "$APP_HOME"

RUN apt update
RUN apt install -y python3-pip python3-wheel

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY *.py ./
COPY services ./services
COPY data ./data
COPY interfaces ./interfaces

EXPOSE $PORT

CMD ["gunicorn","--config", "gunicorn_config.py", "app:app"]