FROM python:latest
WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir
RUN mkdir data
COPY server.py /server
ENTRYPOINT ["python3"]
CMD ["server.py"]

