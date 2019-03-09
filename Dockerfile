FROM diego1q2w/python-3.7-local-scientific

WORKDIR /usr/src/app

COPY . .

CMD python main.py
