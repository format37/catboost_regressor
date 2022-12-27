FROM python:latest
WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt --no-cache-dir
# RUN python3 -m pip install -U scikit-learn scipy matplotlib
COPY server.py /server
ENTRYPOINT ["python3"]
CMD ["server.py"]