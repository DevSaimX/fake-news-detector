import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from streamlit_echarts import st_echarts  # ✅ NEW

# -------------------------------
# 🔧 Load model and tokenizer
# -------------------------------

model = load_model('model/fake_news_lstm_model.h5')

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model('model/fake_news_lstm_model.h5')

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


# -------------------------------
# 🧼 Text Cleaning Function
# -------------------------------

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# -------------------------------
# 🤖 Prediction Function
# -------------------------------

def predict_news(news_text):
    cleaned_text = clean_text(news_text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=500)
    prediction = model.predict(padded)[0][0]
    return prediction

# -------------------------------
# 🖥️ Streamlit UI
# -------------------------------

st.set_page_config(page_title="📰 Fake News Detection", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article text to detect whether it's **Fake** or **Real**.")

user_input = st.text_area("✍️ Paste News Article Below:", height=250)

if st.button("🔍 Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        result = predict_news(user_input)
        label = "🔴 Fake News" if result > 0.5 else "🟢 Real News"
        confidence = result if result > 0.5 else 1 - result
        explanation = (
            "This article seems **suspicious** and may contain **fabricated information**."
            if result > 0.5 else
            "This article appears to be **legitimate** and **trustworthy**."
        )

        st.success(f"**Prediction:** {label}")
        st.info(f"🧠 **Confidence Score:** {confidence * 100:.2f}%")
        st.write(f"🗒️ **Explanation:** {explanation}")

        # 📈 Display confidence both as bar chart and gauge meter
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Class Confidence")
            st.bar_chart({
                'Confidence': {
                    'Fake News': float(result),
                    'Real News': 1 - float(result)
                }
            })

        with col2:
            st.subheader("🎯 Prediction Confidence Gauge")
            confidence_percentage = float(confidence * 100)

            option = {
                "series": [
                    {
                        "type": "gauge",
                        "startAngle": 180,
                        "endAngle": 0,
                        "min": 0,
                        "max": 100,
                        "progress": {"show": True, "width": 18},
                        "axisLine": {"lineStyle": {"width": 18}},
                        "pointer": {"show": True},
                        "title": {"show": True, "offsetCenter": [0, "70%"]},
                        "detail": {
                            "valueAnimation": True,
                            "formatter": "{value}%",
                            "offsetCenter": [0, "40%"]
                        },
                        "data": [{"value": round(confidence_percentage, 2), "name": "Confidence"}],
                    }
                ]
            }

            st_echarts(options=option, height="300px")
