FROM python:3.5

ADD . .
RUN pip install numpy emoji text_unidecode sklearn torch torchvision flask
EXPOSE 5005

CMD [ "python", "main.py" ]
