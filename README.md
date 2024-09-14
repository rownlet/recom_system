# Sistema de Recomendación de Productos para Comercios Locales

Este proyecto implementa un sistema de recomendación para comercios locales basado en modelos de similitud como Jaccard, Coseno, y KNN. Utiliza `Streamlit` para la visualización y permite a los usuarios seleccionar una comuna y un comercio para obtener recomendaciones personalizadas de productos.

## Despliegue en la Web

El sistema de recomendaciones está desplegado en la web y puede ser accedido en el siguiente enlace: [Sistema de Recomendación de Productos para Comercios Locales](https://recom-system.onrender.com/).

## Estructura del Proyecto

- **app.py**: El archivo principal que contiene el código de la aplicación Streamlit.
- **commerces.csv**: Datos sobre los comercios locales (ID del comercio y distrito).
- **product.csv**: Información sobre los productos (ID del producto, categoría y precio).
- **transactions.csv**: Datos sobre las transacciones (ID del comercio, ID del producto, cantidad y precio).
- **requirements.txt**: Archivo con las dependencias del proyecto.
- **docs/**: Carpeta con archivos relacionados al desafío técnico.

## Instalación

1. Clonar este repositorio:
    ```
    git clone <url_del_repositorio>
    ```

2. Instalar las dependencias:
    ```
    pip install -r requirements.txt
    ```

3. Ejecutar la aplicación:
    ```
    streamlit run app.py
    ```

## Dependencias

Este proyecto utiliza las siguientes bibliotecas:

- `streamlit==1.35.0`: Para la visualización de la interfaz de usuario.
- `plotly==5.19.0`: Para la visualización interactiva de gráficos.
- `pandas==2.1.1`: Para la manipulación y análisis de datos.
- `scikit-learn==1.3.0`: Para los modelos de machine learning como KNN.
- `ipywidgets==8.1.0`: Para los widgets interactivos dentro de la aplicación.
- `setuptools`: Requerido para la correcta instalación de algunas dependencias.

## Problemas Encontrados y Soluciones

### 1. **ModuleNotFoundError**: 'plotly'
   - Solución: Asegurarse de que `plotly` esté correctamente instalado ejecutando:
     ```
     pip install plotly
     ```

### 2. **ModuleNotFoundError**: 'sklearn'
   - Solución: Instalar la biblioteca `scikit-learn`:
     ```
     pip install scikit-learn
     ```

### 3. **ModuleNotFoundError**: 'ipywidgets'
   - Solución: Instalar la biblioteca `ipywidgets`:
     ```
     pip install ipywidgets
     ```

### 4. **TypeError**: OneHotEncoder `sparse_output` no es reconocido.
   - Solución: Actualizar a la versión más reciente de `scikit-learn`, ya que `sparse_output` reemplaza a `sparse` en versiones recientes.

## Recomendaciones

Si encuentras algún problema al ejecutar el código, verifica que todas las dependencias estén correctamente instaladas y que tu entorno de Python esté actualizado. Puedes ejecutar la aplicación en local utilizando `streamlit run app.py` para visualizar las recomendaciones.

---

### Autor
Este proyecto ha sido realizado como parte de un desafío técnico.