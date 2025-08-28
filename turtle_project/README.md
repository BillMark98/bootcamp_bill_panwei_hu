
# Turtle Trading Strategy Project

**A comprehensive systematic trading strategy implementation with machine learning capabilities**

## 🎯 Project Overview

This project implements the Turtle Trading Strategy, a systematic trend-following approach that combines traditional technical analysis with modern machine learning techniques. The strategy identifies trends using Donchian Channels and moving averages, manages risk with ATR-based position sizing, and generates signals for entry and exit points across multiple asset classes.

## 📊 Key Features

- **Systematic Trading Strategy**: Implements the classic Turtle Trading methodology
- **Machine Learning Integration**: Uses regression and classification models for return prediction
- **Risk Management**: ATR-based position sizing and portfolio allocation
- **Multi-Asset Support**: Diversified universe of ETFs across asset classes
- **Real-time API**: Flask API for model predictions and analysis
- **Interactive Dashboard**: Streamlit interface for exploration and monitoring
- **Comprehensive Documentation**: Stakeholder-ready reports and technical documentation

## 🏗️ Project Structure

```
turtle_project/
├── data/
│   ├── raw/              # Raw data from APIs (CSV format)
│   └── processed/        # Cleaned, validated data (Parquet format)
├── src/
│   ├── config.py         # Configuration and environment setup
│   ├── utils.py          # General utilities
│   ├── preprocessing.py  # Data preprocessing and cleaning
│   ├── feature_engineering.py  # Feature creation and engineering
│   ├── eda_utils.py      # Exploratory data analysis utilities
│   ├── risk_analysis.py  # Risk management and analysis
│   └── model_utils.py    # Model training, prediction, and management
├── notebooks/
│   ├── 00_project_setup.ipynb
│   ├── 01_problem_framing.ipynb
│   ├── 02_tooling_setup.ipynb
│   ├── 03_python_fundamentals.ipynb
│   ├── 04_data_acquisition.ipynb
│   ├── 05_data_storage.ipynb
│   ├── 06_data_preprocessing.ipynb
│   ├── 07_outliers_risk_analysis.ipynb
│   ├── 08_exploratory_data_analysis.ipynb
│   ├── 09_feature_engineering.ipynb
│   ├── 10a_regression_modeling.ipynb
│   ├── 10b_timeseries_modeling.ipynb
│   └── 11_evaluation_risk_communication.ipynb
├── model/                # Saved model files
├── reports/              # Generated reports and outputs
├── deliverables/         # Stakeholder-ready deliverables
├── app.py               # Flask API server
├── app_streamlit.py     # Streamlit dashboard
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- API keys (optional, for live data)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd turtle_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   # Create .env file
   echo "API_KEY=your_api_key_here" > .env
   echo "DATA_DIR_RAW=./data/raw" >> .env
   echo "DATA_DIR_PROCESSED=./data/processed" >> .env
   ```

### Running the Application

#### Option 1: Flask API
```bash
python app.py
```
The API will be available at `http://localhost:5000`

#### Option 2: Streamlit Dashboard
```bash
streamlit run app_streamlit.py
```
The dashboard will be available at `http://localhost:8501`

#### Option 3: Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## 📡 API Documentation

### Base URL
`http://localhost:5000`

### Endpoints

#### 1. Home Page
- **GET** `/`
- **Description**: API documentation and usage examples
- **Response**: HTML page with interactive documentation

#### 2. Health Check
- **GET** `/health`
- **Description**: Check API health and model status
- **Response**: JSON with system status

#### 3. Model Predictions
- **POST** `/predict`
- **Description**: Make predictions using trained models
- **Request Body**:
  ```json
  {
    "features": [100.5, 98.2, 105.1, 95.3, 2.1, 0.025, 0.3, 0.6],
    "model_type": "regression"
  }
  ```
- **Response**:
  ```json
  {
    "predictions": {
      "Ridge": 0.0023,
      "Linear": 0.0018,
      "Random Forest": 0.0031
    },
    "features_used": [100.5, 98.2, 105.1, 95.3, 2.1, 0.025, 0.3, 0.6],
    "timestamp": "2025-08-27T10:30:00"
  }
  ```

#### 4. Sample Predictions
- **GET** `/predict/sample`
- **Description**: Get sample prediction data for testing
- **Response**: JSON with sample data and predictions

#### 5. Full Analysis
- **POST** `/run_full_analysis`
- **Description**: Run complete analysis pipeline
- **Request Body**:
  ```json
  {
    "data_path": "data/processed/turtle_enhanced_features_20250825_140536.parquet",
    "output_dir": "reports"
  }
  ```
- **Response**: JSON with analysis results and output paths

#### 6. Model Summary
- **GET** `/model_summary`
- **Description**: Get summary of loaded models and performance
- **Response**: JSON with model information

#### 7. Reports
- **GET** `/reports`
- **Description**: List available reports and outputs
- **Response**: JSON with file information

- **GET** `/reports/<filename>`
- **Description**: Download specific report file
- **Response**: File download

### Example API Usage

#### Python
```python
import requests
import json

# Make a prediction
url = "http://localhost:5000/predict"
data = {
    "features": [100.5, 98.2, 105.1, 95.3, 2.1, 0.025, 0.3, 0.6],
    "model_type": "regression"
}
response = requests.post(url, json=data)
predictions = response.json()
print(predictions)
```

#### cURL
```bash
# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [100.5, 98.2, 105.1, 95.3, 2.1, 0.025, 0.3, 0.6]}'

# Run full analysis
curl -X POST http://localhost:5000/run_full_analysis \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/processed/turtle_enhanced_features_20250825_140536.parquet"}'
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **📊 Predictions**: Interactive prediction interface with manual input, sample data, and file upload
- **🔬 Analysis**: Full analysis pipeline execution and monitoring
- **📈 Performance**: Model performance tracking and visualization
- **⚙️ Settings**: Configuration and model management

## 📈 Model Performance

### Current Model Performance (Real Data)

Based on our analysis with real financial data:

- **Best R² Score**: -0.0086 (Ridge Regression)
- **Best Directional Accuracy**: 52.8%
- **Best Binary Classification Accuracy**: 52.7%
- **Annual Return Estimate**: 3.5%
- **Annual Volatility**: 30.0%
- **Sharpe Ratio**: 0.007

### Performance Reality Check

⚠️ **Important**: These results reflect typical performance for financial prediction models:
- R² values are typically low or negative due to market efficiency
- Directional accuracy slightly above random (50%) is common
- The value lies in systematic approach and risk management
- Appropriate for 1-2% portfolio allocation with proper risk controls

## 🔧 Development

### Adding New Features

1. **Data Processing**: Add functions to `src/preprocessing.py`
2. **Feature Engineering**: Extend `src/feature_engineering.py`
3. **Model Training**: Use `src/model_utils.py` for model management
4. **API Endpoints**: Add routes to `app.py`
5. **Dashboard**: Extend `app_streamlit.py` with new pages

### Testing

```bash
# Run API tests
python -m pytest tests/

# Run model validation
python -c "from src.model_utils import TurtleModelManager; print('Models working')"
```

## 📋 Stakeholder Handoff Summary

### Project Purpose
This project implements a systematic Turtle Trading strategy enhanced with machine learning capabilities for institutional trading applications.

### Key Findings
1. **Modest Predictive Power**: Models show typical financial prediction performance with R² values around -0.01 to 0.01
2. **Directional Accuracy**: Slightly above random at ~52-53%
3. **Risk Management Value**: Systematic approach provides consistent risk controls
4. **Diversification Benefits**: Multi-asset approach reduces correlation risk

### Recommendations
1. **Conservative Implementation**: Start with 1-2% portfolio allocation
2. **Risk Controls**: Implement 2% daily loss limits and position sizing
3. **Regular Monitoring**: Track performance weekly, retrain models quarterly
4. **Combination Strategy**: Use as part of diversified portfolio approach

### Assumptions and Limitations
- **Historical Data**: Models trained on 2023-2025 data may not reflect future market conditions
- **Market Regimes**: Performance may vary significantly across different market environments
- **Transaction Costs**: Analysis does not include trading costs and slippage
- **Liquidity**: Assumes sufficient liquidity for position entry/exit

### Risks and Issues
- **Model Overfitting**: Risk of overfitting to historical data
- **Regime Change**: Strategy may underperform in changing market conditions
- **Data Quality**: Dependence on data quality and API reliability
- **Implementation Risk**: Real-world execution may differ from backtest results

### Next Steps
1. **Live Testing**: Implement paper trading to validate real-world performance
2. **Risk Monitoring**: Develop automated risk monitoring and alerting systems
3. **Model Enhancement**: Explore ensemble methods and alternative feature sets
4. **Performance Optimization**: Optimize for transaction costs and market impact

## 📚 Documentation

### Technical Reports
- `deliverables/TURTLE_TRADING_EXECUTIVE_SUMMARY.md`: Executive summary
- `deliverables/TURTLE_TRADING_TECHNICAL_REPORT.md`: Detailed technical report
- `deliverables/INVESTMENT_COMMITTEE_MEMO.md`: Investment committee memorandum
- `deliverables/ASSUMPTIONS_AND_RISKS.md`: Risk documentation

### Code Documentation
- `TURTLE_TRADING_GUIDE.md`: Comprehensive strategy guide
- `ENV_SETUP.md`: Environment setup instructions
- `src/`: Well-documented source code with docstrings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Panwei Hu** - *Initial work* - [GitHub Profile]

## 🙏 Acknowledgments

- Richard Dennis and William Eckhardt for the original Turtle Trading concept
- NYU MFE Bootcamp for educational framework
- Open source community for tools and libraries

## 📞 Support

For questions or support:
- Create an issue in the repository
- Contact: [Your Email]
- Documentation: See `docs/` folder for detailed guides

---

**Last Updated**: August 27, 2025  
**Version**: 1.0.0  
**Status**: Production Ready
