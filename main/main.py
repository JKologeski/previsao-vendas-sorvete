# main.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


data = {
    'Temperatura': [22, 25, 28, 30, 32, 35, 37, 40, 42, 45],
    'Vendas': [50, 55, 60, 65, 70, 80, 85, 95, 100, 110]
}


df = pd.DataFrame(data)


X = df[['Temperatura']] 
y = df['Vendas']  


model = LinearRegression()


mlflow.start_run()


model.fit(X, y)


mlflow.sklearn.log_model(model, "modelo_vendas_sorvete")


previsoes = model.predict(X)


plt.scatter(X, y, color='blue', label='Vendas reais')
plt.plot(X, previsoes, color='red', label='Previsões')
plt.xlabel('Temperatura')
plt.ylabel('Vendas')
plt.title('Previsão de Vendas de Sorvete com Base na Temperatura')
plt.legend()
plt.show()


mlflow.end_run()
