### RoBERTa Model for Sentiment Classification

One way to test the code is via POST

```
curl -X POST -H "Content-Type: application/json" -d '{"text": "This movie was absolutely fantastic! I loved every minute of it."}' http://127.0.0.1:8000/predict

```

Or by running webapp/test_app.py
