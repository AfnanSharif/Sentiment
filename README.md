<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=FC466B,3F5EFB&height=200&section=header&text=Sentiment&fontSize=70&fontColor=ffffff&animation=twinkling" width="100%" />

<img src="https://img.icons8.com/?id=43664&format=png&size=100" width="90" />

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=2500&pause=1000&color=FC466B&center=true&vCenter=true&width=700&height=50&lines=Topic%20Modeling%20Analyzer;scikit-learn%20+%20Streamlit;Trainable%20Demo%20Pipeline" alt="Typing SVG" />

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)

</div>

---

## 📖 Overview

**Sentiment** is a topic modeling analyzer scaffold: it trains a scikit-learn/XGBoost model on a
generated sample dataset (`src/data_generator.py` → `src/train.py`) and serves live
predictions through a small Streamlit UI (`src/app.py`).

Dependencies (`requirements.txt`):
```
  pandas>=1.3.0
  numpy>=1.21.0
  scikit-learn>=1.0
  streamlit>=1.20.0
  pyyaml>=6.0
  joblib>=1.2.0
```

## 🏗️ Project Layout

```
Sentiment/
├── configs/config.yaml   # Data paths & model hyperparameters
├── data/external/         # Generated sample dataset
├── models/                 # Trained model artifact (.pkl)
├── notebooks/               # Exploratory notebooks
├── src/
│   ├── app.py                 # Streamlit UI — loads the trained model and predicts
│   ├── train.py               # Trains the model from configs/config.yaml
│   ├── data_generator.py      # Generates the sample dataset
│   ├── features/               # Feature engineering
│   ├── models/                  # Model definitions
│   └── visualization/            # Plotting helpers
└── tests/
```

## ⚡ Setup & Run

### 🪟 Windows (PowerShell / CMD)
```cmd
git clone https://github.com/AfnanSharif/Sentiment.git
cd Sentiment

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src/train.py
streamlit run src/app.py
```

### 🍎 macOS / 🐧 Linux
```bash
git clone https://github.com/AfnanSharif/Sentiment.git
cd Sentiment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/train.py
streamlit run src/app.py
```

Open **http://localhost:8501**.

---

<div align="center">

**Created by [AfnanSharif](https://github.com/AfnanSharif)** · ⭐ star this repo if it helped you

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=FC466B,3F5EFB&height=80&section=footer" width="100%" />

</div>
