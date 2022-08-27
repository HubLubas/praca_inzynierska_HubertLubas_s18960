FROM python:3.10

WORKDIR /usr/src

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/usr/src

COPY assets callbacks/*.py layouts/*.py app.py /usr/src/

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

CMD ["python3", "/usr/src/app.py"]
