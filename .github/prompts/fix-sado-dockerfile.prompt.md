---
description: "Fix the Sado service Dockerfile and docker-compose build config so uvicorn is installed and the container starts correctly"
name: "Fix Sado Dockerfile"
agent: "agent"
---

Fix the Sado service so it builds and starts correctly in Docker. There are two issues:

## Issue 1 — `install_dependency.py` skips installation inside Docker

File: [install_dependency.py](../../../Sado/install_dependency.py)

The `__main__` block calls `install_dep()` only when running inside a virtual environment:
```python
if is_venv():
    install_dep()
else:
    print("Not running inside a virtual environment")
```

Docker builds use the system Python interpreter, not a venv, so `is_venv()` returns `False` and no packages are ever installed. `uvicorn` is missing at runtime, causing the container to crash.

**Fix**: Remove the venv guard and always call `install_dep()`:
```python
if __name__ == "__main__":
    install_dep()
```

## Issue 2 — Wrong `build.context` in docker-compose.yml

File: [docker-compose.yml](../../docker-compose.yml)

The `sado` service has:
```yaml
build:
  context: ../..
  dockerfile: Dockerfile
```

From `d:\Github\DockerCompose`, `../..` resolves to the drive root `d:\`, not `d:\Github\Sado`. The `COPY` instructions in the Dockerfile can't find the source files.

**Fix**: Change the context to point to the Sado project directory:
```yaml
build:
  context: ../Sado
  dockerfile: Dockerfile
```

## After fixing

Rebuild the image without cache and restart:
```bash
docker compose build sado --no-cache
docker compose up
```
