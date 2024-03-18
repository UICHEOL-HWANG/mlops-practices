FROM amd64/python:3.9-slim

WORKDIR /usr/kafka-api

RUN pip install -U pip &&\
    pip install psycopg2-binary kafka-python requests

COPY data_subscriber.py data_subscriber.py

ENTRYPOINT ["python", "data_subscriber.py"]