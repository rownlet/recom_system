import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np

# Cargar los datos (ajusta las rutas según sea necesario)
comercios = pd.read_csv('commerces.csv')  # Contiene: id_commerce, district
productos = pd.read_csv('product.csv')    # Contiene: id_product, category, price
transacciones = pd.read_csv('transactions.csv')  # Contiene: id_commerce, id_product, quantity, price

# Mostrar imagen de Yom
car_logo_url = "https://www.yom.ai/wp-content/uploads/2024/01/Yom_Logo001-1.png"

# Usar Markdown para centrar la imagen horizontalmente
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="{car_logo_url}" alt="Yom Logo" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

# Agregar un contenedor para hacer espacio
with st.container():
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

# Encabezado de la aplicación centrado
st.markdown("<h1 style='text-align: center;'>Sistema de Recomendación de Productos para Comercios Locales</h1>", unsafe_allow_html=True)

# Agregar un contenedor para hacer espacio
with st.container():
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

# Paso 1: Agrupar por id_commerce e id_product, sumando la cantidad comprada para consolidar las transacciones duplicadas
transactions_group = transacciones.groupby(['id_commerce', 'id_product']).agg({'quantity': 'sum', 'price': 'sum'}).reset_index()

# Paso 2: Unir los dataframes de transacciones, comercios y productos
data = transactions_group.merge(comercios, on='id_commerce', how='inner').merge(productos, on='id_product', how='inner')

# Paso 3: Eliminamos la columna 'name' pero conservamos 'id_product'
clean_data = data.drop(columns=['price_x', 'name'])

# Paso 4: Aplicar OneHotEncoder a las columnas 'district' y 'category'
encoder = OneHotEncoder(sparse_output=False)

# Seleccionar las columnas a codificar
columns_to_encode = clean_data[['district', 'category']]

# Ajustar y transformar las columnas
encoded_columns = encoder.fit_transform(columns_to_encode)

# Crear un DataFrame con las columnas codificadas
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['district', 'category']))

# Paso 5: Concatenar el DataFrame original con las nuevas columnas codificadas y eliminar las originales
merged_data_encoded = pd.concat([clean_data.drop(columns=['district', 'category']), encoded_df], axis=1)

# Paso 6: Crear la matriz de compras (comercios vs productos)
purchase_matrix = transactions_group.pivot_table(index='id_commerce', columns='id_product', values='quantity', aggfunc='sum', fill_value=0)

# Normalizar los valores de quantity en la matriz de compras
scaler = MinMaxScaler()
purchase_matrix_scaled = pd.DataFrame(scaler.fit_transform(purchase_matrix), index=purchase_matrix.index, columns=purchase_matrix.columns)

# Widget para seleccionar la comuna
comunas_disponibles = comercios['district'].unique().tolist()
comuna_seleccionada = st.selectbox('Selecciona la comuna', comunas_disponibles)

# Filtrar comercios de la comuna seleccionada
comercios_en_comuna = comercios[comercios['district'] == comuna_seleccionada]['id_commerce'].tolist()

# Widget para seleccionar el comercio
id_commerce_seleccionado = st.selectbox('Selecciona el comercio', comercios_en_comuna)

# Widget para seleccionar el modelo
modelo_seleccionado = st.selectbox('Selecciona el modelo de recomendación', ['Jaccard', 'Coseno', 'KNN'])

# Función para calcular la similitud de Jaccard
def jaccard_similarity(commerce1, commerce2):
    intersection = np.sum(np.minimum(commerce1, commerce2))
    union = np.sum(np.maximum(commerce1, commerce2))
    return intersection / union

# Función para normalizar los scores entre 0 y 1
def normalizar_scores(df, score_column):
    scaler = MinMaxScaler()
    df[score_column] = scaler.fit_transform(df[[score_column]])
    return df

# Función para obtener productos personalizados basados en similitud de Jaccard
def obtener_productos_jaccard(id_commerce, top_n=10):
    if id_commerce not in purchase_matrix.index:
        return f"El comercio con ID {id_commerce} no existe."

    target_commerce = purchase_matrix.loc[id_commerce]
    
    # Calcular la similitud de Jaccard con todos los demás comercios
    similarities = purchase_matrix.apply(lambda x: jaccard_similarity(target_commerce, x), axis=1)
    
    # Excluir el comercio mismo
    similarities = similarities.drop(id_commerce)
    
    # Obtener el comercio más similar
    most_similar_commerce = similarities.idxmax()
    
    # Obtener los productos más comprados por el comercio más similar
    similar_products = purchase_matrix.loc[most_similar_commerce]
    
    return generar_recomendaciones(id_commerce, similar_products, top_n)

# Función para obtener productos personalizados basados en similitud de Coseno
def obtener_productos_coseno(id_commerce, top_n=10):
    if id_commerce not in purchase_matrix.index:
        return f"El comercio con ID {id_commerce} no existe."
    
    # Calcular similitudes de coseno con todos los demás comercios
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(purchase_matrix)
    
    # Encontrar los comercios más cercanos usando la similitud de coseno
    distances, indices = knn.kneighbors([purchase_matrix.loc[id_commerce]], n_neighbors=10)
    
    # Excluir el comercio mismo
    most_similar_commerce = purchase_matrix.index[indices.flatten()[1]]
    
    # Obtener los productos más comprados por el comercio más similar
    similar_products = purchase_matrix.loc[most_similar_commerce]
    
    return generar_recomendaciones(id_commerce, similar_products, top_n)

# Función para obtener productos personalizados basados en KNN
def obtener_productos_knn(id_commerce, top_n=10):
    if id_commerce not in purchase_matrix.index:
        return f"El comercio con ID {id_commerce} no existe."
    
    # Entrenar el modelo KNN con métrica de coseno
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(purchase_matrix)
    
    # Obtener el índice del comercio seleccionado en la matriz
    selected_commerce_index = purchase_matrix.index.get_loc(id_commerce)
    
    # Encontrar los vecinos más cercanos usando KNN
    distances, indices = knn.kneighbors([purchase_matrix.iloc[selected_commerce_index]], n_neighbors=10)
    
    # Obtener los IDs de los comercios similares
    similar_commerces = purchase_matrix.index[indices.flatten()[1:]]  # Excluimos el comercio mismo
    
    # Obtener los productos más comprados por los comercios más similares
    similar_products = purchase_matrix.loc[similar_commerces].sum(axis=0)
    
    return generar_recomendaciones(id_commerce, similar_products, top_n)

# Función para generar recomendaciones de productos
def generar_recomendaciones(id_commerce, similar_products, top_n=10):
    # Obtener los productos comprados por el comercio
    productos_comercio = transacciones[transacciones['id_commerce'] == id_commerce]
    productos_comprados = set(productos_comercio['id_product'].unique())
    
    # En lugar de excluir, ponderamos los productos ya comprados con menor relevancia
    productos_ajustados = productos.copy()  # Copiamos todos los productos
    productos_ajustados['popularity_score'] = productos_ajustados['id_product'].apply(
        lambda x: similar_products[x] if x in similar_products.index else 0
    )
    
    # Si el producto ya ha sido comprado, reducimos su score
    productos_ajustados['adjusted_score'] = productos_ajustados.apply(
        lambda row: row['popularity_score'] * 0.5 if row['id_product'] in productos_comprados else row['popularity_score'],
        axis=1
    )
    
    # Normalizar los scores entre 0 y 1
    productos_ajustados = normalizar_scores(productos_ajustados, 'adjusted_score')
    
    # Ordenar los productos por la popularidad ajustada y seleccionar los top N
    top_recommendations = productos_ajustados.sort_values(by='adjusted_score', ascending=False).head(top_n)
    
    # Devolver las recomendaciones
    return top_recommendations[['id_product', 'adjusted_score', 'category', 'price']]

# Función para mostrar las recomendaciones
if st.button('Mostrar recomendaciones'):
    if modelo_seleccionado == 'Jaccard':
        recomendaciones = obtener_productos_jaccard(id_commerce_seleccionado)
    elif modelo_seleccionado == 'Coseno':
        recomendaciones = obtener_productos_coseno(id_commerce_seleccionado)
    else:
        recomendaciones = obtener_productos_knn(id_commerce_seleccionado)

    # Reiniciar el índice del DataFrame de recomendaciones
    recomendaciones = recomendaciones.reset_index(drop=True)
    recomendaciones.index = recomendaciones.index
    
    st.write(f"Recomendaciones para el comercio {id_commerce_seleccionado}:")
    st.write(recomendaciones)
