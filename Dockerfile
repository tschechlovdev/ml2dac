FROM python:3.9.6
WORKDIR /app
COPY . .
RUN pip install --upgrade pip

COPY . .

RUN apt-get update \
&& apt-get install gcc make g++ -y \
&& apt-get clean


RUN python setup.py bdist_wheel
RUN pip install dist/* --prefer-binary
CMD ["python", "reproducibility.py"]
#CMD ["python", "test.py"]
