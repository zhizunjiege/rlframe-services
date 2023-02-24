FROM tensorflow/tensorflow:2.9.1-gpu
WORKDIR /app
COPY . .
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r agent-requirements.txt
EXPOSE 10002 6006
VOLUME [ "/app/data" ]
ENTRYPOINT [ "python", "agent.py", "-p", "10002" ]
CMD [ "-w", "10", "-m", "256" ]
