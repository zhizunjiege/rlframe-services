FROM python:3.10.4-slim
WORKDIR /app
COPY . .
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r bff-requirements.txt
EXPOSE 10000
VOLUME [ "/app/data" ]
ENTRYPOINT [ "python", "bff.py", "-p", "10000" ]
CMD [ "-w", "10", "-m", "256" ]
