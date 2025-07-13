from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification


app = Flask(__name__)
# Load the tokenizer and the model with a sequence classification head
# `roberta-base` is the pre-trained model checkpoint
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")

labels = ["negative", "positive"]

# (Optional) Move the model to the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval() # Set the model to evaluation mode


@app.route("/predict", methods = ["POST"])

def predict():
    # Make sure the request contains a JSON object
    if not request.json or "text" not in request.json:
        return jsonify({"error": "Invalid JSON payload. Please send a 'text' field."}), 400

    text = request.json["text"]
    
    # Use the tokenizer to prepare the input
    # The `return_tensors="pt"` argument ensures we get PyTorch tensors.
    # `truncation=True` and `padding=True` are good practices for handling input length.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Move the input tensors to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform inference. The `with torch.no_grad():` block is crucial for
    # efficiency during inference as it disables gradient calculations.
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the raw prediction scores (logits) from the model's output
    logits = outputs.logits

    # Find the index of the class with the highest logit value
    predicted_class_id = torch.argmax(logits, dim=1).item()

    # Get the corresponding label from our labels list
    predicted_label = labels[predicted_class_id]
    
    # You can also get the probabilities by applying a softmax function
    probabilities = torch.nn.functional.softmax(logits, dim=1).tolist()[0]
    
    # Return the result as a JSON response
    return jsonify({
        "text_input": text,
        "predicted_label": predicted_label,
        "predicted_class_id": predicted_class_id,
        "probabilities": probabilities
    })

if __name__ == "__main__":
    app.run(host= "0.0.0.0", port = 8000, debug = True)