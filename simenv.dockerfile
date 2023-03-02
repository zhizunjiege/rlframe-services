FROM python:3.10.4-slim
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends g++-mingw-w64-x86-64 && pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r simenv-requirements.txt
EXPOSE 10001
VOLUME [ "/app/data" ]
ENTRYPOINT [ "python", "simenv.py", "-p", "10001" ]
CMD [ "-w", "10", "-m", "4" ]
