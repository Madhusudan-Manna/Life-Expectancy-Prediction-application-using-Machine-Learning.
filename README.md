# 🏥 Life Expectancy Prediction - End-to-End ML Application

A comprehensive Streamlit application for predicting life expectancy using machine learning algorithms.

## 📋 Project Overview

This application demonstrates a complete machine learning workflow:
- **Data Exploration**: Analyze WHO life expectancy dataset
- **Data Preprocessing**: Handle missing values, outliers, transformations
- **Model Training**: Compare 7 different ML algorithms
- **Prediction Interface**: Real-time life expectancy predictions
- **Performance Analytics**: Detailed model evaluation and insights

## 🚀 Quick Start

### Local Installation

1. **Clone/Download the repository**
   ```bash
   cd "Life Expectancy Prediction"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # On Windows PowerShell
   # or
   source .venv/bin/activate   # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will open at `http://localhost:8501`

## 📦 Deployment on Streamlit Cloud

### Prerequisites
- GitHub account with the repository
- Streamlit Cloud account (https://streamlit.io/cloud)

### Steps

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your GitHub repo and branch
   - Set main file path to `app.py`
   - Click "Deploy"

### Important Files for Deployment

- **`requirements.txt`** - Python dependencies (NOT `req.txt`)
- **`.streamlit/config.toml`** - Streamlit configuration
- **`app.py`** - Main Streamlit application
- **`Life Expectancy Data.csv`** - Dataset (required for app to work)

## 📊 Features

### Navigation Pages
1. **📊 Data Overview** - Dataset statistics and summary
2. **🔍 Exploratory Data Analysis** - Visualizations and correlations
3. **⚙️ Model Training & Evaluation** - Train and compare models
4. **🎯 Make Predictions** - Real-time predictions interface
5. **📈 Model Performance** - Detailed performance metrics
6. **ℹ️ About** - Project documentation

## 🤖 Models Included

- Linear Regression
- Ridge & Lasso Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

## 🛠️ Technologies Used

- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning algorithms
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **XGBoost** - Gradient boosting
- **SciPy** - Statistical functions

## 📁 Project Structure

```
Life Expectancy Prediction/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── Life Expectancy Data.csv        # Dataset
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── .gitignore                      # Git ignore file
└── best_life_expectancy_model.pkl # Trained model (auto-generated)
```

## ⚠️ Troubleshooting

### Error: "File not found: Life Expectancy Data.csv"
- Ensure the dataset file is in the same directory as `app.py`
- Check file name spelling (case-sensitive)

### Error: "No module named 'sklearn'"
- Run: `pip install -r requirements.txt`
- Ensure you're using the correct virtual environment

### Deployment Error: "Cannot import sklearn"
- **Ensure `requirements.txt` exists** (not `req.txt`)
- Redeploy the app after fixing requirements.txt
- Clear Streamlit Cloud cache if needed

## 🔄 Common Issues During Deployment

### Issue: Import errors on Streamlit Cloud
**Solution**: Use `requirements.txt` (NOT `req.txt`)

Common mistakes:
❌ `req.txt` - Wrong filename
✅ `requirements.txt` - Correct filename

### Issue: GridSearchCV import error
**Solution**: Removed unused imports from app.py

## 📈 Expected Results

- **Best Model**: Usually Random Forest or XGBoost
- **Typical R² Score**: 0.70 - 0.85
- **Prediction Accuracy**: ± 3-5 years

## 📝 License

This project is for educational purposes.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure dataset file exists
4. Check Streamlit version compatibility

---

**Built with ❤️ for educational and demonstration purposes**
