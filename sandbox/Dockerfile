FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update


RUN apt install -y python3.10 python3.10-venv python3.10-dev git curl wget

RUN ls -la /usr/bin/python3 && rm /usr/bin/python3 && ln -s python3.10 /usr/bin/python3

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3 -m pip --version

# chdir to /app
WORKDIR /app

# 安装python package
COPY requirements.txt requirements.txt
RUN pip install -U -r requirements.txt

COPY . /app

CMD ["gunicorn", "-w", "8", "-b", "0.0.0.0:5000", "app:app"]

# gunicorn -w 8 -b 0.0.0.0:5000 app:app
