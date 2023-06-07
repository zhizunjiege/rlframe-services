FROM tensorflow/tensorflow:2.9.1-gpu
WORKDIR /app
COPY . .
ENV TZ=Asia/Shanghai
RUN apt-get update \
  && apt-get -y install procps \
  && pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r agent-requirements.txt \
  && chmod +x agent.sh
EXPOSE 6006 10002
VOLUME [ "/app/data" ]
ENTRYPOINT [ "./agent.sh" ]
CMD [ "-w", "10", "-m", "256", "-l", "info" ]
