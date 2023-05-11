FROM python:3.10.4-slim
WORKDIR /app
COPY . .
ENV TZ=Asia/Shanghai
RUN apt-get update \
  && apt-get install -y --no-install-recommends g++-mingw-w64-x86-64 procps \
  && pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r simenv-requirements.txt \
  && chmod +x simenv.sh
EXPOSE 10001
VOLUME [ "/app/data" ]
ENTRYPOINT [ "./simenv.sh" ]
CMD [ "-w", "10", "-m", "4" ]
