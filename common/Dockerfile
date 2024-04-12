FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY  ./main.py /code/app/
COPY  ./models.py /code/app/
COPY  ./index.html /code/Templates/
COPY  ./styles.css /code/static/
COPY  ./JetBrainsMono-Medium.ttf /code/static


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
