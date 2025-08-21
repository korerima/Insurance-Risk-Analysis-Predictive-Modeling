"""
Command-line interface for insurance risk analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.config.settings import get_settings, create_directories
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator
from src.models.risk_predictor import RiskPredictor


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("insurance_risk_analysis.log")
        ]
    )


def validate_data(args) -> None:
    """Validate data quality."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data validation...")
    
    settings = get_settings()
    create_directories(settings)
    
    # Load data
    loader = DataLoader(settings)
    try:
        data = loader.load_data(args.data_file)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Validate data
    validator = DataValidator()
    result = validator.validate_data(data)
    
    # Print validation report
    print(validator.generate_validation_report(result))
    
    if not result.is_valid:
        logger.error("Data validation failed!")
        sys.exit(1)
    
    logger.info("Data validation completed successfully!")


def preprocess_data(args) -> None:
    """Preprocess and clean data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing...")
    
    settings = get_settings()
    create_directories(settings)
    
    # Load data
    loader = DataLoader(settings)
    try:
        data = loader.load_data(args.input_file)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Preprocess data
    preprocessor = DataPreprocessor(settings)
    
    # Clean data
    cleaned_data = preprocessor.clean_data(data)
    logger.info(f"Data cleaning completed. Shape: {cleaned_data.shape}")
    
    # Engineer features
    engineered_data = preprocessor.engineer_features(cleaned_data)
    logger.info(f"Feature engineering completed. Shape: {engineered_data.shape}")
    
    # Save processed data
    output_file = args.output_file or "processed_data.csv"
    loader.save_data(engineered_data, output_file)
    
    logger.info(f"Data preprocessing completed! Output saved to: {output_file}")


def train_model(args) -> None:
    """Train a risk prediction model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    settings = get_settings()
    create_directories(settings)
    
    # Load and preprocess data
    loader = DataLoader(settings)
    preprocessor = DataPreprocessor(settings)
    
    try:
        data = loader.load_data(args.data_file)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Preprocess data
    cleaned_data = preprocessor.clean_data(data)
    engineered_data = preprocessor.engineer_features(cleaned_data)
    
    # Prepare features
    X, feature_names = preprocessor.prepare_features(engineered_data)
    y = engineered_data['HadClaim']
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Train model
    model = RiskPredictor(
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.training_metrics['accuracy']
    test_predictions = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    test_score = accuracy_score(y_test, test_predictions)
    
    logger.info(f"Model training completed!")
    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    
    # Save model
    model_file = args.model_file or f"risk_model_{args.model_type}.pkl"
    model.save_model(model_file)
    logger.info(f"Model saved to: {model_file}")
    
    # Print feature importance if available
    try:
        importance = model.get_feature_importance()
        if importance:
            print("\nTop 10 Feature Importance:")
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, score in sorted_features:
                print(f"  {feature}: {score:.4f}")
    except NotImplementedError:
        logger.info("Feature importance not available for this model type")


def predict_risk(args) -> None:
    """Make risk predictions on new data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting risk prediction...")
    
    settings = get_settings()
    
    # Load model
    model = RiskPredictor()
    try:
        model.load_model(args.model_file)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load and preprocess data
    loader = DataLoader(settings)
    preprocessor = DataPreprocessor(settings)
    
    try:
        data = loader.load_data(args.data_file)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Preprocess data
    cleaned_data = preprocessor.clean_data(data)
    engineered_data = preprocessor.engineer_features(cleaned_data)
    X, _ = preprocessor.prepare_features(engineered_data)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_risk_probability(X)
    risk_categories = model.get_risk_categories(X)
    
    # Add predictions to data
    result_data = engineered_data.copy()
    result_data['RiskPrediction'] = predictions
    result_data['RiskProbability'] = probabilities
    result_data['RiskCategory'] = risk_categories
    
    # Save results
    output_file = args.output_file or "risk_predictions.csv"
    loader.save_data(result_data, output_file)
    
    logger.info(f"Risk prediction completed! Results saved to: {output_file}")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"  Total records: {len(result_data)}")
    print(f"  High risk: {(result_data['RiskCategory'] == 'High').sum()}")
    print(f"  Very high risk: {(result_data['RiskCategory'] == 'Very High').sum()}")
    print(f"  Average risk probability: {probabilities.mean():.4f}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Insurance Risk Analysis - Predictive Modeling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate data quality
  python -m src.cli validate --data-file data.csv
  
  # Preprocess data
  python -m src.cli preprocess --input-file raw_data.csv --output-file processed_data.csv
  
  # Train a model
  python -m src.cli train --data-file processed_data.csv --model-type random_forest
  
  # Make predictions
  python -m src.cli predict --data-file new_data.csv --model-file model.pkl
        """
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data quality")
    validate_parser.add_argument("--data-file", required=True, help="Path to data file")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess and clean data")
    preprocess_parser.add_argument("--input-file", required=True, help="Path to input data file")
    preprocess_parser.add_argument("--output-file", help="Path to output file (default: processed_data.csv)")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a risk prediction model")
    train_parser.add_argument("--data-file", required=True, help="Path to training data file")
    train_parser.add_argument("--model-type", default="random_forest", 
                             choices=["random_forest", "gradient_boosting", "logistic", "svm"],
                             help="Type of model to train (default: random_forest)")
    train_parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators (default: 100)")
    train_parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth (default: 10)")
    train_parser.add_argument("--model-file", help="Path to save model (default: auto-generated)")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make risk predictions")
    predict_parser.add_argument("--data-file", required=True, help="Path to data file for predictions")
    predict_parser.add_argument("--model-file", required=True, help="Path to trained model file")
    predict_parser.add_argument("--output-file", help="Path to output file (default: risk_predictions.csv)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    try:
        if args.command == "validate":
            validate_data(args)
        elif args.command == "preprocess":
            preprocess_data(args)
        elif args.command == "train":
            train_model(args)
        elif args.command == "predict":
            predict_risk(args)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
