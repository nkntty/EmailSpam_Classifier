## Automated Email Spam Ditector Model
The Problem: "Inboxes are flooded with phishing and junk mail, reducing productivity."
The Solution: "Built a classification model using Scikit-Learn that identifies spam with [Insert your accuracy, e.g., 98%] accuracy."
How to run: "Install dependencies using pip install -r requirements.txt and run Task2_Python_script.py."



# Email Classifier Demo (Streamlit)

A Streamlit app that demonstrates multiple spam-detection models (PyTorch) trained on email feature vectors.

## Features
- Select a model from the sidebar
- Upload a CSV of email features
- Get predictions (Spam/Ham) + confidence

## Project Structure
├── demo.py
├── models/
│ ├── danylo.pth
│ ├── aitua.pth
│ ├── leshawn.pth
│ ├── tatsuya.pth
│ └── esther.pth
├── sample_data/
│ └── example.csv
├── requirements.txt
└── .gitignore



## Setup (Local)
### 1) Clone the repo
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_FOLDER>


2) Create and activate a virtual environment
macOS / Linux
----------------------------
python3 -m venv .venv
source .venv/bin/activate
----------------------------

Windows (PowerShell)
----------------------------
python -m venv .venv
.\.venv\Scripts\Activate.ps1
----------------------------

3) Install dependencies
------------------------------
pip install -r requirements.txt
------------------------------
(If torch installation fails, install PyTorch using the official instructions for your system (CPU/GPU):
https://pytorch.org/get-started/locally/)

4) Run the app by typing the following command on the same directory, including "demo.py":
------------------------------
streamlit run demo.py
------------------------------

