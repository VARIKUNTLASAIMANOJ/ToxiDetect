ToxiDetect: Advanced Comment Toxicity Detection System
ToxiDetect is a deep learning-powered solution for identifying and classifying toxic comments across multiple online platforms. By leveraging state-of-the-art natural language processing (NLP) models, ToxiDetect ensures safer and more inclusive digital spaces through accurate detection and classification of harmful content.

📋 Features
Toxicity Classification: Detects and categorizes comments into Low, Medium, and High toxicity levels.
Multi-Input Support: Processes text, audio (via speech-to-text), and image inputs (using OCR).
Transformer-Based Model: Uses fine-tuned DistilBERT for enhanced contextual understanding.
Real-Time Feedback Dashboard: Visualizes user feedback and system performance metrics.
User-Friendly Interface: Built using Gradio for intuitive and interactive user interaction.
Continuous Learning: Feedback integration for model refinement over time.
🎯 Objectives
Develop a high-accuracy classification model for toxicity detection.
Incorporate multiple input formats for diverse use cases.
Ensure user accessibility through an interactive web application.
Provide actionable insights with advanced visualization tools.
Enable ethical and fair detection while minimizing unintended biases.
🚀 Technologies Used
Software Stack
Programming Language: Python (v3.8+)
Libraries:
NLP & ML: TensorFlow/Keras, Hugging Face Transformers
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Text Processing: NLTK, TextBlob
Web Interface: Gradio
OCR & Speech: Pytesseract, SpeechRecognition
Hardware Requirements
Processor: Multi-core CPU (Intel i5 or equivalent)
RAM: 8 GB minimum
GPU: NVIDIA GTX 1060 or higher for training acceleration
Storage: 10 GB for datasets and models
🧪 Model Overview
Architecture:

Preprocessing: Tokenization, normalization, slang expansion
Feature Extraction: DistilBERT embeddings
Classification: LSTM for context analysis
Output: Toxicity category probabilities
Metrics:

Accuracy, Precision, Recall, and F1-score for model evaluation.
Confusion Matrix for detailed analysis of predictions.
Dataset:

Comments from Wikipedia talk pages, labeled for various toxicity types (e.g., threats, insults, obscenities).
💻 Deployment and Usage
Local Setup
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/ToxiDetect.git
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the application:
bash
Copy
Edit
python app.py
Access the web interface at http://127.0.0.1:7887.
Web Application
Upload text, audio, or image comments.
View real-time toxicity predictions and visualizations.
Provide feedback to improve the model's accuracy.
📊 Results
High Accuracy: Robust classification of toxic comments.
Visualization: Real-time display of prediction probabilities and trends.

🛠️ Future Enhancements
Multilingual Support: Extend functionality for non-English languages.
Advanced Models: Incorporate larger models like GPT for nuanced understanding.
API Integration: Provide an API for external applications.
Enhanced Visualization: Advanced dashboards for better insights.
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

👩‍💻 Contributors
Varikutla Sai Manoj
Sravya Vasa
Veeramalla Sai Charan
Vemula Sravani
Vemuri Pranay
Under the guidance of Prof. K. Manoj Sagar.

