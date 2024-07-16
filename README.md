# FASTAPI-BE

## Create venv
```bash
python -m venv mayday
```

## Change Python Dev Env
```bash
source ./mayday/Scripts/activate
pip install -r requirements.txt
```

## Start Server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```