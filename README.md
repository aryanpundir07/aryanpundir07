# Fake vs True News Classifier

## Train the model

1) Install deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Train
```bash
python train_model.py --true_csv /path/to/true.csv --fake_csv /path/to/fake.csv --model_out model.joblib
```

## Run the web app

Set the `MODEL_PATH` env var if your model is not `./model.joblib`.

```bash
export MODEL_PATH=${MODEL_PATH:-model.joblib}
flask --app app:app run --host 0.0.0.0 --port 5000
# or production
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

### API

```bash
curl -s -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Sample news paragraph here"}' | jq
```

- 👋 Hi, I’m @aryanpundir07
- 👀 I’m interested in learning programming and new skills.
- 🌱 I’m currently learning languages C++ and Java.
- 📫 LinkedIn: www.linkedin.com/in/aryan-pundir-92a556291
      Email: aryanpundir2021@gmail.com 

