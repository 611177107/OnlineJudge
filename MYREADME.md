# do this when update backend

```bash
docker build -t 611177107/fan_onlinejudge:dev . && docker push 611177107/fan_onlinejudge:dev
```

# do this after image updated
```bash
docker-compose down && docker-compose pull && docker-compose up -d
```