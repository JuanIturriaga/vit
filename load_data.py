import pandas as pd
import numpy as np
import tensorflow as tf
import os

def load_dataframe_xray(dataset_info_path = './ds/NIH_Chest_X-ray_1024_info'):
    # Función específica para cargar el dataset de rayos X de NIH
    # La función devuelve los DataFrames de entrenamiento y prueba 
    # con las rutas de las imágenes y las etiquetas binarias.

    # Cargar las rutas absolutas y crear un diccionario de mapeo
    with open(os.path.join(dataset_info_path, 'dir.txt'), 'r') as f: #[cite: 1]
        rutas_absolutas = f.read().splitlines()

    # Asignar cada nombre de archivo a su ruta completa
    mapa_rutas = {os.path.basename(ruta): ruta for ruta in rutas_absolutas}

    # Cargar metadatos
    df = pd.read_csv(os.path.join(dataset_info_path, 'Data_Entry_2017.csv'))
    df['Path'] = df['Image Index'].map(mapa_rutas)

    # Descartar filas de imágenes que no estén físicamente en el directorio
    df = df.dropna(subset=['Path'])

    # Leer las listas de partición
    with open(os.path.join(dataset_info_path, 'train_val_list.txt'), 'r') as f: 
        train_files = f.read().splitlines()

    with open(os.path.join(dataset_info_path,'test_list.txt'), 'r') as f: 
        test_files = f.read().splitlines()

    # Generar columnas binarias para cada etiqueta posible
    etiquetas = df['Finding Labels'].str.get_dummies(sep='|')   

    # Genera etiquetas solo para 2 categorías, de este modo queda balanceado
    etiquetas['Patology'] = np.where(etiquetas['No Finding'] == 1, 0, 1)
    etiquetas = etiquetas.drop(columns=[col for col in etiquetas.columns if col not in ['No Finding', 'Patology']])

    # Nombres de las clases finales (No Finding y Patology)
    nombres_clases = etiquetas.columns.tolist()
    print("Nombres de las clases finales:")
    print(nombres_clases)

    # Concatenar las etiquetas codificadas al DataFrame principal
    df = pd.concat([df, etiquetas], axis=1)

    # Dividir el DataFrame final en Train y Test
    train_df = df[df['Image Index'].isin(train_files)]
    test_df = df[df['Image Index'].isin(test_files)]

    return train_df, test_df


def plot_label_distribution(df, class_names):
    import matplotlib.pyplot as plt
    etiquetas = df[class_names]
    cantidad = etiquetas.sum().sort_values(ascending=False)
    cantidad.plot(kind='bar')
    plt.show()


if __name__ == "__main__":

    nombres_clases = ['No Finding', 'Patology']
    train_df, test_df = load_dataframe_xray()

    print(f"\nTrain DataFrame {train_df.shape}:")
    print(train_df.head())
    plot_label_distribution(train_df, nombres_clases)

    print(f"\nTest DataFrame {test_df.shape}:")
    print(test_df.head())
    plot_label_distribution(test_df, nombres_clases)
