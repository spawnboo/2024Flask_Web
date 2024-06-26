FROM python:3.8.18-slim

WORKDIR /Flask_Web
ADD . /Flask_Web
COPY . /Flask_Web

RUN pip install -r requirements.txt

EXPOSE 5000

CMD python main.py