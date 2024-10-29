FROM python:3.7.16

WORKDIR /2024Flask_Web
ADD . /2024Flask_Web
COPY . /2024Flask_Web

RUN pip install -r requirements.txt

EXPOSE 80

CMD python FlaskWeb.py