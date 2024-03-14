FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install torch psycopg2-binary torchvision

COPY create_mnist_table.py insert_data.py run_scripts.sh ./

# run_scripts.sh 실행 권한 부여
RUN chmod +x run_scripts.sh

CMD ["./run_scripts.sh"]
# inster_data.py 내용 -> run_scripts.sh
# 누락 부분 재수정 푸시