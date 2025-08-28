from flask import Flask, render_template, request
import re
import emoji
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import LabelEncoder
import json
import os

# --- Flask app ---
app = Flask(__name__)

# --- Paths to models/tokenizers ---
base_path = r"E:/self_emotion/model"
self_tokenizer_path = os.path.join(base_path, "bert_tokenizer")
emotion_tokenizer_path = os.path.join(base_path, "bert_emotion_tokenizer")
self_model_path = os.path.join(base_path, "bert_self_reference_model_saved")
emotion_model_path = os.path.join(base_path, "bert_emotion_model_saved")
label_encoder_path = os.path.join(base_path, "label_encoder.json")  

# --- Load tokenizers ---
self_tokenizer = BertTokenizer.from_pretrained(self_tokenizer_path)
emotion_tokenizer = BertTokenizer.from_pretrained(emotion_tokenizer_path)

# --- Load models (SavedModel format) ---
self_model = tf.keras.models.load_model(self_model_path)
emotion_model = tf.keras.models.load_model(emotion_model_path)

# --- Load label encoder from JSON ---
with open(label_encoder_path, "r") as f:
    classes = json.load(f)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(classes)

# --- Text preprocessing ---
def clean_texts(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Prediction function ---
def predict_tweet(tweet):
    cleaned_tweet = clean_texts(tweet)

    # --- Self-reference prediction ---
    encoding_self = self_tokenizer(
        [cleaned_tweet],
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors="tf"
    )

    # Use SavedModel signature for inference
    infer_self = self_model.signatures["serving_default"]
    outputs_self = infer_self(
        input_ids=encoding_self["input_ids"],
        attention_mask=encoding_self["attention_mask"]
    )
    prob_self = list(outputs_self.values())[0][0][0]  # first sample

    is_self = prob_self >= 0.5
    result = {
        "tweet": tweet,
        "self_reference": "Yes" if is_self else "No",
        "self_reference_score": f"{prob_self*100:.2f}%"
    }

    # --- Emotion prediction only if self-referential ---
    if is_self:
        encoding_emotion = emotion_tokenizer(
            [cleaned_tweet],
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="tf"
        )

        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        bert_model.trainable = False
        bert_outputs = bert_model(
            input_ids=encoding_emotion["input_ids"],
            attention_mask=encoding_emotion["attention_mask"]
        )
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]

        # Emotion model inference using signature
        infer_emotion = emotion_model.signatures["serving_default"]
        outputs_emotion = infer_emotion(cls_embedding)
        probs = list(outputs_emotion.values())[0].numpy()
        pred_idx = np.argmax(probs, axis=1)[0]
        confidence = probs[0][pred_idx]*100
        label = label_encoder.inverse_transform([pred_idx])[0]

        result.update({
            "emotion": label,
            "emotion_confidence": f"{confidence:.2f}%"
        })
    else:
        result.update({
            "emotion": "Skipped",
            "emotion_confidence": "0%"
        })

    return result

# --- Flask routes ---
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        tweet = request.form.get("tweet")
        if tweet:
            prediction = predict_tweet(tweet)
    return render_template("index.html", prediction=prediction)

# --- Run server ---
if __name__ == "__main__":
    app.run(debug=True)
