import tensorflow as tf
import numpy as np

# Hiperparámetros base
input_shape = (72, 72, 3) # Imágenes redimensionadas
patch_size = 6            # Tamaño de cada parche (6x6)
num_patches = (input_shape[0] // patch_size) ** 2 # Total: 144 parches
projection_dim = 64       # Dimensión del vector de cada token de imagen
num_heads = 4
transformer_layers = 4
num_classes = 2

# 1. Capa para extraer parches de la imagen original
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # Extraer parches usando operaciones tensoriales (sin convoluciones de aprendizaje)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        # Aplanar los parches en una secuencia 1D
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

# 2. Capa de Proyección Lineal y Codificación Posicional
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        # Proyección lineal (Embedding)
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        # Vector de posición aprendible
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # Sumar la proyección del parche con su posición en la grilla original
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.position_embedding.output_dim,
        })
        return config

# 3. Construcción del modelo ViT completo
def create_vit_classifier():
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Preprocesamiento (ej. Data Augmentation podría ir aquí)
    
    # Crear parches
    patches = Patches(patch_size)(inputs)
    # Codificar parches (Proyección + Posición)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Bloques del Transformer (Self-Attention)
    for _ in range(transformer_layers):
        # Normalización y Atención
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Conexión residual
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        
        # Feed Forward Network (MLP)
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = tf.keras.layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = tf.keras.layers.Dropout(0.1)(x3)
        x3 = tf.keras.layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)
        # Conexión residual
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Representación final
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # En lugar del token [CLS] clásico, aquí usamos GlobalAveragePooling
    # para colapsar la secuencia en un único vector
    representation = tf.keras.layers.GlobalAveragePooling1D()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    
    # Capa de clasificación final
    features = tf.keras.layers.Dense(1024, activation=tf.nn.gelu)(representation)
    features = tf.keras.layers.Dropout(0.5)(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

# 4. Instanciar y compilar

vit_model = create_vit_classifier()
vit_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

vit_model.summary()


# cargar dataset 
from load_data import load_dataframe_xray
from dataset_builder import ImagePreprocessor, DatasetBuilder


nombres_clases = ['No Finding', 'Patology']
train_df, test_df = load_dataframe_xray(dataset_info_path = './ds/NIH_Chest_X-ray_1024_info') 
# El dataframe trae una columna "path" con la ruta y de la imagen y columnas con las clases son: 'No Finding', 'Patology'
# Por lo tanto requiere un preprocesador que lea la columna "path" y convierta las clases en etiquetas binarias.


# Definir el preprocesador y construir los datasets
mobilenet_preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
preprocessor = ImagePreprocessor(
    img_size= (input_shape[0], input_shape[1]),
    preprocess_fn=mobilenet_preprocess
)
dataset_builder = DatasetBuilder(
    preprocessor=preprocessor,
    batch_size=32
)
train_ds = dataset_builder.build(train_df, nombres_clases)
test_ds = dataset_builder.build(test_df, nombres_clases)


history = vit_model.fit(
    train_ds,
    epochs=50,
    validation_data=test_ds
)

# PASO 4: Guardar los resultados del experimento

# Busca un id al experimento basado en el prefijo y las capetas existentes en resultados
import os

RESULTS_PATH = './results'
EXPERIMENT_PREFIX = 'XRAY_VIT_'
IMG_SIZE = (input_shape[0], input_shape[1])
EPOCHS = 50

os.makedirs(RESULTS_PATH, exist_ok=True)

existing_experiments = [d for d in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, d))]
experiment_ids = [int(d.replace(EXPERIMENT_PREFIX, '')) for d in existing_experiments if d.startswith(EXPERIMENT_PREFIX) and d.replace(EXPERIMENT_PREFIX, '').isdigit()]
next_experiment_id = max(experiment_ids, default=0) + 1
experiment_id = f"{EXPERIMENT_PREFIX}{next_experiment_id:04}"
experiment_path = os.path.join(RESULTS_PATH, experiment_id)
os.makedirs(experiment_path, exist_ok=True)

# guarda parámetros del experimento en json 
params = {
    'img_size': IMG_SIZE,
    'train_batch_size': 32,
    'train_epochs': EPOCHS,
    'train_learning_rate': 1e-3,
    'experiment_id': experiment_id,
    'experiment_prefix': EXPERIMENT_PREFIX,
    'results_path': RESULTS_PATH,
    'model_transfer': 'MobileNetV2',
    'pretrained_weights': 'imagenet',
    'num_classes': len(nombres_clases)
}
import json
with open(os.path.join(experiment_path, 'experiment_params.json'), 'w') as f:
    json.dump(params, f)


# Guardar el modelo entrenado
vit_model.save(os.path.join(experiment_path, 'mobilenet_xray_model.keras'))

# Guardar el historial de entrenamiento en csv
import csv
with open(os.path.join(experiment_path, 'training_history.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    # Escribir los encabezados
    headers = list(history.history.keys())
    writer.writerow(headers)
    # Escribir los valores por cada epoch
    for i in range(len(history.history[headers[0]])):
        row = [history.history[h][i] for h in headers]
        writer.writerow(row)


# Guardar los resultados de la evaluación en csv
eval_results = vit_model.evaluate(test_ds)
with open(os.path.join(experiment_path, 'evaluation_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    for name, value in zip(vit_model.metrics_names, eval_results):
        writer.writerow([name, value])



