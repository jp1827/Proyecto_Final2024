from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.cluster import KMeans
import pandas as pd
import pyodbc

app = Flask(__name__)

# Conexión a la base de datos SQL Server 2022
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=JOSEM\\SQL2022;DATABASE=sistemaExperto;UID=sa;PWD=Gotrade18')
cursor = conn.cursor()

# Cargar el modelo entrenado
modelo = joblib.load('modelo_decision_tree.pkl')

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    nuevos_datos = [
        float(data['ingreso_mensual']) if data['ingreso_mensual'] else 0, 
        float(data.get('valor_propiedad', 0)) if data.get('valor_propiedad') else 0, 
        float(data.get('monto_solicitado', 0)) if data.get('monto_solicitado') else 0, 
        float(data.get('cantidad_deudas', 0)) if data.get('cantidad_deudas') else 0, 
        float(data.get('costo_curso', 0)) if data.get('costo_curso') else 0, 
        float(data.get('valor_activos_negocio', 0)) if data.get('valor_activos_negocio') else 0, 
        float(data.get('costo_reformas', 0)) if data.get('costo_reformas') else 0, 
        float(data.get('costo_viaje', 0)) if data.get('costo_viaje') else 0
    ]
    
    # Realizar la predicción
    tipo_prestamo_predicho = modelo.predict([nuevos_datos])[0]
    
    return jsonify({'tipo_prestamo_predicho': tipo_prestamo_predicho})

@app.route('/generar_reglas', methods=['POST'])
def generar_reglas():
    nuevas_reglas = generar_nuevas_reglas()
    insertar_nuevas_reglas(nuevas_reglas)
    return jsonify({'message': 'Nuevas reglas generadas e insertadas con éxito.'})

@app.route('/verificar_reglas', methods=['POST'])
def verificar_reglas():
    data = request.json
    # Convertir datos de entrada a float
    datos_usuario = {
        'ingreso_mensual': float(data['ingreso_mensual']) if data['ingreso_mensual'] else 0,
        'valor_propiedad': float(data.get('valor_propiedad', 0)) if data.get('valor_propiedad') else 0,
        'monto_solicitado': float(data.get('monto_solicitado', 0)) if data.get('monto_solicitado') else 0,
        'cantidad_deudas': float(data.get('cantidad_deudas', 0)) if data.get('cantidad_deudas') else 0,
        'costo_curso': float(data.get('costo_curso', 0)) if data.get('costo_curso') else 0,
        'valor_activos_negocio': float(data.get('valor_activos_negocio', 0)) if data.get('valor_activos_negocio') else 0,
        'costo_reformas': float(data.get('costo_reformas', 0)) if data.get('costo_reformas') else 0,
        'costo_viaje': float(data.get('costo_viaje', 0)) if data.get('costo_viaje') else 0
    }
    conclusiones = encadenamiento_hacia_adelante(datos_usuario, data['tipo_prestamo'])
    return jsonify({'conclusiones': conclusiones})

def cargar_datos_no_procesados():
    query = "SELECT * FROM datos_no_procesados"
    df = pd.read_sql(query, conn)
    return df

def cargar_reglas():
    cursor.execute("SELECT * FROM reglas")
    reglas = cursor.fetchall()
    return reglas

def cargar_conclusiones():
    cursor.execute("SELECT * FROM conclusiones")
    conclusiones = cursor.fetchall()
    return conclusiones

def normalizar_nombre(nombre):
    if nombre is None:
        return ''
    return nombre.lower().replace('préstamo de ', 'préstamo ').replace('préstamo ', '').replace('tarjeta de crédito', 'tarjetas de crédito').replace('emergencias', 'emergencia').replace('para vacaciones', 'vacaciones').strip()

def evaluar_reglas(datos_usuario, reglas, tipo_prestamo_seleccionado):
    resultados = []
    tipo_prestamo_seleccionado_normalizado = normalizar_nombre(tipo_prestamo_seleccionado)
    for regla in reglas:
        id_regla, tipo_prestamo, ingreso_mensual_minimo, valor_propiedad_minimo, monto_maximo_solicitado, cantidad_maxima_deudas, costo_maximo_curso, valor_activos_negocio_minimo, costo_maximo_reformas, costo_maximo_viaje = regla
        if tipo_prestamo is not None and normalizar_nombre(tipo_prestamo) == tipo_prestamo_seleccionado_normalizado and (
            datos_usuario.get('ingreso_mensual', 0) >= ingreso_mensual_minimo and
            (valor_propiedad_minimo is None or datos_usuario.get('valor_propiedad', 0) >= valor_propiedad_minimo) and
            (monto_maximo_solicitado is None or datos_usuario.get('monto_solicitado', 0) <= monto_maximo_solicitado) and
            (cantidad_maxima_deudas is None or datos_usuario.get('cantidad_deudas', 0) <= cantidad_maxima_deudas) and
            (costo_maximo_curso is None or datos_usuario.get('costo_curso', 0) <= costo_maximo_curso) and
            (valor_activos_negocio_minimo is None or datos_usuario.get('valor_activos_negocio', 0) >= valor_activos_negocio_minimo) and
            (costo_maximo_reformas is None or datos_usuario.get('costo_reformas', 0) <= costo_maximo_reformas) and
            (costo_maximo_viaje is None or datos_usuario.get('costo_viaje', 0) <= costo_maximo_viaje)):
            resultados.append(tipo_prestamo)
    return resultados


def encadenamiento_hacia_adelante(datos_usuario, tipo_prestamo_seleccionado):
    reglas = cargar_reglas()
    conclusiones = cargar_conclusiones()
    prestamos_posibles = evaluar_reglas(datos_usuario, reglas, tipo_prestamo_seleccionado)
    
    resultados = []
    tipo_prestamo_seleccionado_normalizado = normalizar_nombre(tipo_prestamo_seleccionado)
    for prestamo in prestamos_posibles:
        if normalizar_nombre(prestamo) == tipo_prestamo_seleccionado_normalizado:
            for conclusion in conclusiones:
                if normalizar_nombre(conclusion[1]) == tipo_prestamo_seleccionado_normalizado:
                    if conclusion[2] not in resultados:
                        resultados.append(conclusion[2])
    return resultados

def generar_nuevas_reglas():
    datos_no_procesados = cargar_datos_no_procesados()
    datos_no_procesados.fillna(0, inplace=True)
    
    # Seleccionar características relevantes para clustering
    X = datos_no_procesados.drop(columns=['id', 'tipo_prestamo'])
    
    # Aplicar K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    
    # Obtener los centroides de los clusters
    centroides = kmeans.cluster_centers_
    
    nuevas_reglas = []
    for centroide in centroides:
        nueva_regla = {
            'tipo_prestamo': 'Nuevo Tipo',
            'ingreso_mensual_minimo': centroide[0],
            'valor_propiedad_minimo': centroide[1],
            'monto_maximo_solicitado': centroide[2],
            'cantidad_maxima_deudas': centroide[3],
            'costo_maximo_curso': centroide[4],
            'valor_activos_negocio_minimo': centroide[5],
            'costo_maximo_reformas': centroide[6],
            'costo_maximo_viaje': centroide[7]
        }
        nuevas_reglas.append(nueva_regla)
    
    return nuevas_reglas

def insertar_nuevas_reglas(nuevas_reglas):
    for regla in nuevas_reglas:
        cursor.execute("""
            INSERT INTO reglas (tipo_prestamo, ingreso_mensual_minimo, valor_propiedad_minimo, monto_maximo_solicitado, cantidad_maxima_deudas, costo_maximo_curso, valor_activos_negocio_minimo, costo_maximo_reformas, costo_maximo_viaje)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, regla['tipo_prestamo'],
           regla['ingreso_mensual_minimo'],
           regla['valor_propiedad_minimo'],
           regla['monto_maximo_solicitado'],
           regla['cantidad_maxima_deudas'],
           regla['costo_maximo_curso'],
           regla['valor_activos_negocio_minimo'],
           regla['costo_maximo_reformas'],
           regla['costo_maximo_viaje'])
    conn.commit()

if __name__ == '__main__':
    app.run(debug=True)