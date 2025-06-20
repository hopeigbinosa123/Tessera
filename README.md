![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Streamlit-orange)
![Build](https://img.shields.io/badge/Build-Stable-brightgreen)

# Tessera: AI NLP Companion

**Tessera** is a modular and beginner-friendly Natural Language Processing (NLP) toolkit built with Python. Designed to simplify and visualize language tasks, Tessera is powered by Hugging Face Transformers, Streamlit, TextBlob, NLTK, and more.

---

## 🚀 Features

- **Sentiment Analysis** — via BERT and TextBlob
- **Text Summarization** — using DistilBART
- **Named Entity Recognition (NER)** — with entity highlighting
- **Question Answering** — extractive answers from context
- **Naive Bayes Classification** — for basic text categorization
- **Tokenization & Preprocessing** — includes stopword removal, stemming, and lemmatization
- **Visualizations** — word clouds and word frequency charts
- **Streamlit Interface** — for interactive multi-task NLP
- **Modular Codebase** — import functions or build your own NLP tools

---

## 📦 Installation

```bash
git clone https://github.com/hopeigbinosa123/tessera.git
cd tessera
python -m venv tessera_env
source tessera_env/bin/activate    # On Windows: .\tessera_env\Scripts\activate
pip install -r requirements.txt
```

---

## 🖥️ Run the Streamlit App

```bash
streamlit run app/tessera_app.py
```

---

## 🧪 Try It Out with Jupyter

You can also test each module interactively using the `demo/examples.ipynb` notebook:
```bash
jupyter notebook demo/examples.ipynb
```

---

## 📁 Folder Structure

```
tessera/
├── app/
│   ├── tessera_app.py               # Streamlit frontend
│   ├── tessera_*.py                 # All NLP modules
├── demo/
│   └── examples.ipynb               # Notebook walkthrough
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🤝 Contributing

Pull requests are welcome! Whether it's adding features, improving performance, or refining documentation—Tessera grows stronger with the community. The application is not perfect. 

---

## 📄 License

This project is licensed under the MIT License.

Tessera is your modular AI-powered toolkit for understanding, summarizing, and analyzing language—fast.
```

