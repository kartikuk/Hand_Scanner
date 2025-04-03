# Hand Scanner Application
A machine learning-based hand scanner for classification and real-time detection.

---

## 🛠 Installation Steps
Follow these steps to set up and run the application.

### ❶1️⃣ Clone the Repository
```bash
git clone https://github.com/kartikuk/Hand_Scanner.git
cd Hand_Scanner
```

### ❶2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate     # Windows
```

### ❶3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application
### ❶1️⃣ Preprocess the Data
```bash
python preprocess.py
```
This script processes and prepares the dataset.

### ❶2️⃣ Train the Model
```bash
python train.py
```
This will train the hand scanner model.

### ❶5️⃣ Run Real-Time Detection
```bash
python real_time_hand_scan.py
```
This starts the real-time scanning process.

---

## 💂️‍♂️ Project Structure
```
Hand_Scanner/
│── augmented_data/        # Augmented dataset images
│── dataset/               # Original dataset images
│── models/                # Trained models
│   └── hand_scan_model.h5 # (Not included in GitHub)
│── train_data/            # Training and validation data
│── app.py                 # Main application file (if applicable)
│── classify_image.py      # Script for classifying a single image
│── evaluate.py            # Model evaluation script
│── labels.csv             # Class labels for the dataset
│── preprocess.py          # Preprocessing script
│── README.md              # Project documentation
│── real_time_hand_scan.py # Real-time hand detection script
│── requirements.txt       # Required Python packages
│── split_data.py          # Splits dataset into train/validation
│── train.py               # Model training script
```

---

## ❗ Notes
- Ensure you have **Python 3.8+** installed.
- If the `.h5` model file is missing, train the model first using `train.py` or download it from the provided source.
- ⚠️This project needs Hardware devices like hand scanning sensors and UV sensors to make this project fully working!!
---

### 📩 Contributions & Issues
Feel free to contribute! If you encounter any issues, open a GitHub issue. 🚀
