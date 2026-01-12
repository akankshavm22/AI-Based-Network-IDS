# 🛡️ AI-Based Network Intrusion Detection System (AI-NIDS)
An Advanced AI-powered Network Intrusion Detection System built using Machine Learning (Random Forest) and Explainable AI via Groq LLM, designed as a student cybersecurity project.
This system detects malicious network traffic from the CIC-IDS dataset, visualizes attack behavior, and provides human-readable security explanations like a virtual SOC analyst.

## Features
- Machine Learning Based Intrusion Detection
  - Random Forest Classifier
  - Balanced class handling
  - High accuracy on network traffic data
- CIC-IDS Dataset Support
  - CSV upload via Streamlit UI
  - Automatic cleaning (NaN, infinite values)
- Live Packet Simulation
  - Random packet capture from test data
  - Real-time attack prediction
- Feature Importance Visualization
  - Displays top contributing network features
  - Helps understand attack behavior
- Explainable AI (Groq LLM Integration)
  - SOC-style explanation of detected attacks
  - Severity assessment
  - Key feature reasoning
  - Suggested mitigation steps
- Model Performance Dashboard
  - Accuracy score
  - Classification report
  - Confusion matrix

 ## Tech Stack 
 
| Category            | Technology                              |
|---------------------|------------------------------------------|
| Frontend            | Streamlit                                |
| ML Model            | Random Forest (scikit-learn)             |
| Data Handling       | Pandas, NumPy                            |
| Visualization       | Matplotlib, Streamlit Charts             |
| Model Persistence   | Joblib                                   |
| Explainable AI      | Groq API (LLaMA 3.3 – 70B)                |

## 📂 Project Structure
<pre>
AI-NIDS/
│
├── app.py                 # Main Streamlit application
├── nids_model.pkl         # Trained Random Forest model (generated)
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
</pre>

## 📊 Dataset
- Dataset Used: CIC-IDS (Canadian Institute for Cybersecurity)
- Format: CSV
- Target Column: Label
- Selected Features:
  - Flow Duration
  - Total Fwd Packets
  - Total Backward Packets
  - Total Length of Fwd Packets
  - Fwd Packet Length Max
  - Flow IAT Mean
  - Flow IAT Std
  - Flow Packets/s

## ⚙️ Installation & Setup
  1️⃣ Clone the Repository
  <pre>
  git clone https://github.com/your-username/AI-NIDS.git
  cd AI-NIDS
</pre>
 
  2️⃣ Install Dependencies
  <pre>
    pip install -r requirements.txt
</pre>

  3️⃣ Run the Application
  <pre>
    streamlit run app.py
</pre>

## 🔑 Groq API Setup (Optional but Recommended)
To enable AI-based explanations:
1. Create a Groq account
2. Generate an API key
3. Paste the key into the Streamlit sidebar
Without the API key, detection still works — only AI explanations are disabled.

## 🧪 How It Works
1. Upload CIC-IDS CSV dataset
2. Click Train Model
3. Model is trained and evaluated
4. Capture a random network packet
5. System predicts:
   - BENIGN or ATTACK
   - Confidence score
6. View:
   - Feature importance
   - AI-generated explanation
   - Model metrics
  
## 📈 Model Details
- Algorithm: Random Forest Classifier
- Trees: 200
- Max Depth: 15
- Class Weight: Balanced
- Test Split: 25%
- Stratified Sampling: Yes

## 🎓 Educational Value
This project demonstrates:
- Practical Network Security Analytics
- Real-world Intrusion Detection
- Explainable AI in Cybersecurity
- Integration of LLMs with ML models
- SOC-style threat analysis

## ⭐ Future Enhancements
- Deep Learning models (LSTM / Autoencoders)
- Real-time packet capture (Wireshark / Scapy)
- SHAP-based explainability
- Web deployment (Docker / Cloud)
- Attack severity scoring engine

## 👩‍💻 Author
Akanksha Mane </br>
Cybersecurity Advanced Project | BE IT

