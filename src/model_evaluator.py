from sklearn.metrics import accuracy_score
from src.logger import logger

class ModelEvalutor:
     def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test,preds)
        logger.info(f"accuracy score: {accuracy}")
        return accuracy