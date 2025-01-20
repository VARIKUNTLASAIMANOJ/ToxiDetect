# ToxiDetect: Advanced Comment Toxicity Detection System

**ToxiDetect** is a deep learning-powered solution for identifying and classifying toxic comments across multiple online platforms. By leveraging state-of-the-art natural language processing (NLP) models, ToxiDetect ensures safer and more inclusive digital spaces through accurate detection and classification of harmful content.

---

## 📋 Features

- **Toxicity Classification**: Detects and categorizes comments into Low, Medium, and High toxicity levels.
- **Multi-Input Support**: Processes text, audio (via speech-to-text), and image inputs (using OCR).
- **Transformer-Based Model**: Uses fine-tuned DistilBERT for enhanced contextual understanding.
- **Real-Time Feedback Dashboard**: Visualizes user feedback and system performance metrics.
- **User-Friendly Interface**: Built using Gradio for intuitive and interactive user interaction.
- **Continuous Learning**: Feedback integration for model refinement over time.

---

## 🎯 Objectives

1. Develop a high-accuracy classification model for toxicity detection.
2. Incorporate multiple input formats for diverse use cases.
3. Ensure user accessibility through an interactive web application.
4. Provide actionable insights with advanced visualization tools.
5. Enable ethical and fair detection while minimizing unintended biases.

---

## 🚀 Technologies Used

### **Software Stack**
- **Programming Language**: Python (v3.8+)
- **Libraries**:
  - **NLP & ML**: TensorFlow/Keras, Hugging Face Transformers
  - **Data Handling**: Pandas, NumPy
  - **Visualization**: Matplotlib, Seaborn
  - **Text Processing**: NLTK, TextBlob
  - **Web Interface**: Gradio
  - **OCR & Speech**: Pytesseract, SpeechRecognition

### **Hardware Requirements**
- **Processor**: Multi-core CPU (Intel i5 or equivalent)
- **RAM**: 8 GB minimum
- **GPU**: NVIDIA GTX 1060 or higher for training acceleration
- **Storage**: 10 GB for datasets and models

---

## 🧪 Model Overview

1. **Architecture**:
   - Preprocessing: Tokenization, normalization, slang expansion
   - Feature Extraction: DistilBERT embeddings
   - Classification: LSTM for context analysis
   - Output: Toxicity category probabilities

2. **Metrics**:
   - Accuracy, Precision, Recall, and F1-score for model evaluation.
   - Confusion Matrix for detailed analysis of predictions.

3. **Dataset**:
   - Comments from Wikipedia talk pages, labeled for various toxicity types (e.g., threats, insults, obscenities).

---

## 💻 Deployment and Usage

### **Local Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ToxiDetect.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Access the web interface at `http://127.0.0.1:7887`.

### **Web Application**
- Upload text, audio, or image comments.
- View real-time toxicity predictions and visualizations.
- Provide feedback to improve the model's accuracy.

---

## 📊 Results

- **High Accuracy**: Robust classification of toxic comments.
- **Visualization**: Real-time display of prediction probabilities and trends.

![GUI Screenshot](path/to/screenshot.png)

---

## 🛠️ Future Enhancements

1. **Multilingual Support**: Extend functionality for non-English languages.
2. **Advanced Models**: Incorporate larger models like GPT for nuanced understanding.
3. **API Integration**: Provide an API for external applications.
4. **Enhanced Visualization**: Advanced dashboards for better insights.

---

## 👩‍💻 Contributors

- ***Varikuntla Sai Manoj***
- ***Sravya Vasa***
- ***Veeramalla Sai Charan***
- ***Vemula Sravani***
- ***Vemuri Pranay***

Under the guidance of **Prof. K. Manoj Sagar**.

---
