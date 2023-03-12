FROM python:3.10.4-slim
WORKDIR /app
COPY . .
RUN apt-get update \
  && apt-get -y install curl procps \
  && curl https://func-e.io/install.sh | bash -s -- -b /usr/local/bin \
  && pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r center-requirements.txt \
  && chmod +x center.sh
EXPOSE 8080 9999 10000
VOLUME [ "/app/data" ]
ENTRYPOINT [ "./center.sh" ]
CMD [ "-w", "10", "-m", "256" ]
