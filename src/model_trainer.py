from sklearn.model_selection import train_test_split
from src.logger import logger
from sklearn.linear_model import LogisticRegression
import pickle


class ModelTraining:
    def train(self, x, y):
        logger.info("model training is started")


        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)


        model = LogisticRegression()
        model.fit(X_train,y_train)


        with open('models/model.pkl', 'wb') as file:
            pickle.dump(model, file)



        logger.info("model training is completed")
        return model, X_test, y_test