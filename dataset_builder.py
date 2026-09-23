import tensorflow as tf
import numpy as np

class ImagePreprocessor:
    """Clase callable para cargar y preprocesar imágenes en pipelines de tf.data."""
    
    def __init__(self, img_size=(224, 224), preprocess_fn=None):
        self.img_size = img_size
        # Permite inyectar la función de preprocesamiento específica del modelo
        self.preprocess_fn = preprocess_fn 

    def __call__(self, path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        
        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)
            
        return img, label

class DatasetBuilder:
    """Orquestador para construir el pipeline de datos."""
    
    def __init__(self, preprocessor, batch_size=32):
        self.preprocessor = preprocessor
        self.batch_size = batch_size

    def build(self, dataframe, class_names):
        paths = dataframe['Path'].values
        labels = dataframe[class_names].values.astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(self.preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


if __name__ == "__main__":

    # --- Ejemplo de uso ---
    import load_data
    train_df, test_df = load_data.load_dataframe_xray()

    # 1. Definir la función de preprocesamiento del modelo base
    mobilenet_preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    # 2. Instanciar el preprocesador con sus hiperparámetros
    preprocessor = ImagePreprocessor(
        img_size=(224, 224), 
        preprocess_fn=mobilenet_preprocess
    )

    # 3. Instanciar el constructor del dataset
    dataset_builder = DatasetBuilder(
        preprocessor=preprocessor, 
        batch_size=32
    )

    # 4. Generar los pipelines
    nombres_clases = ['No Finding', 'Patology']
    train_ds = dataset_builder.build(train_df, nombres_clases)
    test_ds = dataset_builder.build(test_df, nombres_clases)


    print(f"\nTrain Dataset:")
    for images, labels in train_ds.take(1):
        print(images.shape, labels.shape)

    print(f"\nTest Dataset:")
    for images, labels in test_ds.take(1):
        print(images.shape, labels.shape)