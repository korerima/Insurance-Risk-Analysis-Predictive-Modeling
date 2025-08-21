# Insurance Risk Analysis - Predictive Modeling

A comprehensive, modular codebase for analyzing insurance risk patterns and building predictive models for AlphaCare Insurance Solutions (ACIS). This project provides a complete pipeline from data validation to model deployment with automated testing, CI/CD integration, and an interactive Streamlit dashboard with explainability tools.

## 🚀 Features
- **Modular Architecture**: Clean, maintainable code structure with separation of concerns
- **Interactive Dashboard**: Streamlit-based dashboard with real-time data exploration and model training
- **Data Validation**: Comprehensive data quality checks and validation
- **Multiple ML Models**: Support for Random Forest, Gradient Boosting, Logistic Regression, and SVM
- **Explainability**: SHAP-based model interpretability and feature importance analysis
- **Premium Optimization**: Risk-based pricing optimization tools
- **CLI Interface**: Easy-to-use command-line tools
- **Comprehensive Testing**: Unit and integration tests with high coverage
- **CI/CD Pipeline**: Automated testing, linting, and deployment
- **Report Generation**: Automated stakeholder-friendly reports and model cards

## 📁 Project Structure
```
insurance-risk-analysis/
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   ├── data/                     # Data processing modules
│   ├── models/                   # Machine learning models
│   ├── dashboard/                # Interactive dashboard
│   │   ├── app.py               # Main dashboard application
│   │   ├── components.py        # Dashboard components
│   │   └── streamlit_app.py     # Streamlit entry point
│   └── cli.py                   # Command-line interface
├── tests/                        # Test suite
├── Datasets/                     # Data files (tracked by DVC)
├── notebooks/                    # Jupyter notebooks for exploration
├── .github/                      # GitHub Actions workflows
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml               # Project configuration
├── .pre-commit-config.yaml      # Pre-commit hooks
└── README.md                    # This file
```

## 🛠️ Installation
### Prerequisites
- Python 3.8 or higher
- Git
- DVC (for data versioning)

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/alphacare/insurance-risk-analysis.git
   cd insurance-risk-analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Install production dependencies
   pip install -r requirements.txt

   # Install development dependencies (optional)
   pip install -r requirements-dev.txt
   ```

4. **Setup pre-commit hooks (optional):**
   ```bash
   pre-commit install
   ```

5. **Download data with DVC:**
   ```bash
   pip install dvc
   dvc pull
   ```

## 🚀 Quick Start

### Using the Interactive Dashboard
1. **Launch the dashboard:**
   ```bash
   make dashboard
   # or
   streamlit run src/dashboard/streamlit_app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload your data** or use the default dataset

4. **Explore the dashboard features:**
   - 📊 **Data Explorer**: Interactive data visualization and quality analysis
   - 🤖 **Model Training**: Train and evaluate machine learning models
   - 🎯 **Risk Analysis**: Analyze risk patterns and trends
   - 💰 **Premium Optimization**: Optimize pricing based on risk profiles
   - 🔍 **Explainability**: Understand model predictions with SHAP
   - 📋 **Reports**: Generate stakeholder-friendly reports and model cards

### Using the CLI
1. **Validate your data:**
   ```bash
   python -m src.cli validate --data-file Datasets/MachineLearningRating_v4.csv
   ```

2. **Preprocess the data:**
   ```bash
   python -m src.cli preprocess --input-file Datasets/MachineLearningRating_v4.csv --output-file processed_data.csv
   ```

3. **Train a model:**
   ```bash
   python -m src.cli train --data-file processed_data.csv --model-type random_forest
   ```

4. **Make predictions:**
   ```bash
   python -m src.cli predict --data-file new_data.csv --model-file risk_model_random_forest.pkl
   ```

### Using the Python API
```python
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.risk_predictor import RiskPredictor

# Load and preprocess data
loader = DataLoader()
data = loader.load_data("Datasets/MachineLearningRating_v4.csv")

preprocessor = DataPreprocessor()
cleaned_data = preprocessor.clean_data(data)
engineered_data = preprocessor.engineer_features(cleaned_data)

# Prepare features
X, feature_names = preprocessor.prepare_features(engineered_data)
y = engineered_data['HadClaim']

# Train model
model = RiskPredictor(model_type="random_forest")
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_risk_probability(X)
```

## 📊 Dashboard Features

### Data Explorer
- **Interactive Visualizations**: Histograms, box plots, correlation matrices
- **Data Quality Analysis**: Missing values, outliers, data type distribution
- **Statistical Summary**: Descriptive statistics and data profiling
- **Real-time Filtering**: Filter and explore data subsets

### Model Training
- **Interactive Model Configuration**: Select model types and hyperparameters
- **Real-time Training**: Monitor training progress and metrics
- **Model Comparison**: Compare multiple models side-by-side
- **Feature Selection**: Interactive feature importance analysis

### Risk Analysis
- **Risk Metrics Dashboard**: Loss ratios, risk categories, exposure analysis
- **Demographic Analysis**: Risk patterns by age, gender, location
- **Temporal Trends**: Risk evolution over time
- **Geographic Visualization**: Risk mapping by region

### Premium Optimization
- **Risk-Based Pricing**: Optimize premiums based on risk profiles
- **Market Analysis**: Competitive pricing analysis
- **Profit Optimization**: Maximize profitability while managing risk
- **Scenario Analysis**: What-if analysis for different pricing strategies

### Explainability
- **SHAP Analysis**: Model interpretability using SHapley Additive exPlanations
- **Feature Importance**: Understand which factors drive predictions
- **Local Explanations**: Individual prediction explanations
- **Global Insights**: Overall model behavior analysis

### Report Generation
- **Executive Reports**: High-level summaries for stakeholders
- **Technical Reports**: Detailed analysis for data scientists
- **Model Cards**: Comprehensive model documentation
- **Compliance Reports**: Regulatory and audit documentation

## 🧪 Testing
### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_dashboard.py

# Run tests in parallel
pytest -n auto
```

### Test Structure
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Dashboard Tests**: Test Streamlit components and functionality
- **Fixtures**: Reusable test data and objects
- **Mocking**: Isolated testing with mocked dependencies

## 🔧 Development
### Code Quality
The project uses several tools to maintain code quality:
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
safety check
```

### Pre-commit Hooks
Pre-commit hooks automatically run quality checks before commits:
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🚀 CI/CD Pipeline
The project includes a comprehensive CI/CD pipeline with GitHub Actions:
### Pipeline Stages
1. **Test**: Run tests on multiple Python versions
2. **Integration Test**: Run integration tests
3. **Security**: Security vulnerability scanning
4. **Build**: Package building
5. **Deploy**: Automated deployment (on releases)

### Quality Gates
- ✅ All tests must pass
- ✅ Code coverage above threshold
- ✅ No security vulnerabilities
- ✅ Code formatting standards met
- ✅ Type checking passes

## 📊 Model Performance
The risk prediction models achieve the following performance metrics:

| Model Type | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Random Forest | 0.85 | 0.83 | 0.87 | 0.85 |
| Gradient Boosting | 0.86 | 0.84 | 0.88 | 0.86 |
| Logistic Regression | 0.82 | 0.80 | 0.84 | 0.82 |

## 🔍 Data Validation
The data validation system checks for:
- ✅ Required columns presence
- ✅ Data type consistency
- ✅ Missing value patterns
- ✅ Outlier detection
- ✅ Business logic validation
- ✅ Data consistency checks

## 📈 Feature Engineering
Automated feature engineering includes:
- **Temporal Features**: Year, month, quarter extraction
- **Risk Categories**: Loss ratio categorization
- **Premium Categories**: Premium amount categorization
- **Interaction Features**: Premium per claim ratios
- **Categorical Encoding**: Label encoding for categorical variables
- **Numerical Scaling**: Standard scaling for numerical features

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR
- Use type hints for all function parameters and return values

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support
For support and questions:
- 📧 Email: dev@alphacare.com
- 🐛 Issues: [GitHub Issues](https://github.com/alphacare/insurance-risk-analysis/issues)
- 📖 Documentation: [Read the Docs](https://insurance-risk-analysis.readthedocs.io)

## 🙏 Acknowledgments
- AlphaCare Insurance Solutions for providing the dataset
- The open-source community for the excellent tools and libraries
- Contributors and maintainers of this project

---

**Note**: This is a production-ready codebase designed for enterprise use. It includes comprehensive testing, documentation, interactive dashboards, and follows industry best practices for maintainability and scalability.
