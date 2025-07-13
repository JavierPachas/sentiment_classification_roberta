FROM python:3.9

COPY ./requirements.txt /webapp/requirements.txt 

WORKDIR /webapp 

RUN pip install -r requirements.txt

COPY webapp/* /webapp

ENTRYPOINT ["python"]

CMD ["app.py"]