# rlframe-services

## Protos

Use below command to build protos:

For Linux:

```bash
./tools/gen-protos.sh
```

For Windows:

```powershell
./tools/gen-protos.bat
```

## Tests

Use below command to run tests:

For Linux:

```bash
./tools/run-tests.sh
```

For Windows:

```powershell
./tools/run-tests.bat
```

## Build

Use below commands to build independent docker images:

```bash
docker build -f deploy/docker/agent/agent.dockerfile -t rlframe-agent .
docker build -f deploy/docker/simenv/simenv.dockerfile -t rlframe-simenv .
docker build -f deploy/docker/center.dockerfile -t rlframe-center .
```

Or use below command to build an all-in-one docker image:

```
docker build -f deploy/docker/allinone.dockerfile -t rlframe-allinone .
```
