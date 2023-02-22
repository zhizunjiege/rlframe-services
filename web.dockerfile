FROM python:3.10.4-slim
WORKDIR /app
COPY . .
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r web-requirements.txt
EXPOSE 10000
VOLUME [ "/app/data" ]
ENTRYPOINT [ "gunicorn", "-w", "4", "-b", "0.0.0.0:10000", "web:app" ]
