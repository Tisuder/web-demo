FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN mkdir /code/static
RUN mkdir /code/app
RUN mkdir /code/Templates


COPY  ./main.py /code/app
COPY  ./models.py /code/app
COPY  ./index.html /code/Templates
COPY  ./styles.css /code/static
COPY  ./JetBrainsMono-Medium.ttf /code/static/fonts


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
