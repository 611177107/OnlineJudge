# do this when update backend

```bash
docker build -t 611177107/fan_onlinejudge:dev . && docker push 611177107/fan_onlinejudge:dev
```

```bash
docker build -t 611177107/fan_onlinejudge:latest . && docker push 611177107/fan_onlinejudge:latest
```
# do this after image updated
```bash
docker-compose down && docker-compose pull && docker-compose up -d
```

# do this after frontend updated
```bash
rm -r ./OnlineJudgeDeploy/data/backend/dist
ls ./OnlineJudgeDeploy/data/backend/
cp -r ./OnlineJudgeFE/dist OnlineJudgeDeploy/data/backend/
ls ./OnlineJudgeDeploy/data/backend/
```