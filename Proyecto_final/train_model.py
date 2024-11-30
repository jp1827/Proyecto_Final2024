import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Conexión a la base de datos SQL Server 2022
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=JOSEM\\SQL2022;DATABASE=sistemaExperto;UID=sa;PWD=Gotrade18')
cursor = conn.cursor()

# Cargar datos no procesados
def cargar_datos_no_procesados():
    query = "SELECT * FROM datos_no_procesados"
    df = pd.read_sql(query, conn)
    return df

# Función para entrenar el modelo
def entrenar_modelo():
    datos_no_procesados = cargar_datos_no_procesados()
    datos_no_procesados.fillna(0, inplace=True)

    X = datos_no_procesados.drop(columns=['id', 'tipo_prestamo'])
    y = datos_no_procesados['tipo_prestamo']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision}")

    joblib.dump(modelo, 'modelo_decision_tree.pkl')

# Entrenar el modelo y guardarlo en el archivo
if __name__ == "__main__":
    entrenar_modelo()
