FROM node:18.16.0 AS ui
WORKDIR /app
RUN git clone https://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@github.com/zhizunjiege/rlframe-ui.git ui \
  && cd ui \
  && npm install \
  && npm run build

FROM python:3.10.4-slim
WORKDIR /app
COPY . .
COPY --from=ui /app/ui/dist /app/static
ENV TZ=Asia/Shanghai
RUN apt-get update \
  && apt-get -y install curl procps \
  && curl https://func-e.io/install.sh | bash -s -- -b /usr/local/bin \
  && pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r center-requirements.txt \
  && chmod +x center.sh
EXPOSE 8888 9999 10000
VOLUME [ "/app/data" ]
ENTRYPOINT [ "./center.sh" ]
CMD [ "-w", "10", "-m", "256", "-l", "info" ]
