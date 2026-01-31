from src.data_preprocessing import DataPreprocess
from src.model_trainer import ModelTraining
from src.model_evaluator import ModelEvalutor
from src.logger import logger


if __name__ == "__main__":
    logger.info("Sonar Mine Predication")

    X , y = DataPreprocess().preprocess('data/Copy of sonar data.csv')
    model, X_test, y_test = ModelTraining().train(X, y)
    accuracy = ModelEvalutor().evaluate(model, X_test, y_test)

    logger.info(f"pipeline completed with accuracy score {accuracy}")
