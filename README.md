# 🧠 SmartFactCheckBot
AI-powered Fake News Detection using a lightweight Student model distilled from a high-accuracy Teacher model.

SmartFactCheckBot predicts whether a news headline or short article is **REAL** or **FAKE**, and provides the probability for each.  
This project is designed for public benefit and as an open-source contribution to misinformation detection.

---

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
