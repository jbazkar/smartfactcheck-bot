# 🧠 SmartFactCheckBot
AI-powered Fake News Detection using a lightweight Student model distilled from a high-accuracy Teacher model.

SmartFactCheckBot predicts whether a news headline or short article is **REAL** or **FAKE**, and provides the probability for each.  
This project is designed for public benefit and as an open-source contribution to misinformation detection.

---
## ⚠️ Research Limitations
SmartFactCheckBot is an experimental research system designed to analyze linguistic patterns associated with misinformation.
Like all machine-learning research tools, it operates within several important limitations:
## 1. Dataset Constraints
The model is trained on historical misinformation datasets (approx. 2015–2020).
As a result:

It does not reflect modern writing styles or evolving misinformation tactics.
Recent or breaking news may be misclassified.
## 2. No Real-Time Fact Checking

The system does not access live news, APIs, or search engines.

Predictions rely only on:

Linguistic patterns
Writing tone
Statistical signals

Thus:

A true story written in a sensational tone might be flagged as FAKE.
A fake story written in a calm, journalistic tone may be labeled REAL.
The model inherits the biases and limitations of the dataset
## 3. Language & Generalization Limits

The model is trained only on English and may not generalize well to:

Other languages
Cultural writing variations
Highly technical or scientific articles
Satire, sarcasm, or ambiguous text
## 4. Architectural Trade-offs (Distilled Model)

The student model is a smaller, faster version of a larger teacher model.
This design brings trade-offs:

Loss of nuance
Reduced contextual understanding
Increased false positives/negatives
Limitations with complex reasoning
## 5. Ethical and Responsible Use

This system is intended for:

Research
Education
Public awareness
Demonstrating misinformation detection techniques

It must not be used for:

Journalism
Legal decisions
Election monitoring
Crisis response
Safety-critical applications
## No Continual Learning

The model does not update itself automatically.
Changes in:

Political narratives
Social trends
Misinformation strategies
News reporting styles

will affect performance over time.
## 🚀 Features
- Fast, distilled **Student model** for real-time predictions  
- High-accuracy **Teacher model** (DistilBERT fine-tuned on Fake/True News dataset)  
- Simple **CLI interface** for testing  
- Outputs include **REAL / FAKE** label + probabilities  
- Full training scripts included (Teacher + Knowledge Distillation)

---

## 📦 Installation
```bash
git clone https://github.com/jbazkar/smartfactcheckbot.git
cd smartfactcheckbot
pip install -r requirements.txt
```

---

## 📥 Dataset

This project uses the Fake and Real News dataset from Kaggle:

https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

---
## 🧪 Test the Models
```
Test Teacher Model
python training/teacher/test_teacher.py

Test Student Model
python training/student/test_student.py
```

Example:
```
> Trump on Twitter (Dec 29) – Approval rating, Amazon
Prediction: REAL
Probabilities → FAKE: 0.073, REAL: 0.927
```
---

## 🏋️‍♂️ Train the Models
```
Train the Teacher
python training/teacher/train_teacher.py

Knowledge Distillation (Train Student)
python training/student/train_student_kd.py

The student model becomes:
Smaller
Faster
Close to teacher accuracy
```
---
## 📁 Project Structure (Simplified)
```
smartfactcheckbot/
├─ training/
│  ├─ teacher/
│  └─ student/
├─ outputs/
│  ├─ teacher-fast-distilbert/
│  └─ student-distilled/
├─ data/
├─ requirements.txt
└─ README.md
```
## ⚖️ License

MIT License.

## ⭐ Acknowledgements

Teacher and student models built using:

HuggingFace Transformers

PyTorch

Dataset from:

Kaggle Fake/Real News dataset

## 🙌 Contributions

Contributions are welcome. Please submit an issue or pull request.

If you find this project useful, please ⭐ star the repository.
