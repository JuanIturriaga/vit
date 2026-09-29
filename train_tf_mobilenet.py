from dataset_builder import ImagePreprocessor, DatasetBuilder

import tensorflow as tf
from load_data import load_dataframe_xray

IMG_SIZE = (224, 224)
RESULTS_PATH = './results'
EXPERIMENT_PREFIX = 'XRAY_CLA_'
EPOCHS = 800

# PASO 1: Cargar y preprocesar los datos
# Cargar y preprocesar los datos
nombres_clases = ['No Finding', 'Patology']
train_df, test_df = load_dataframe_xray(dataset_info_path = './ds/NIH_Chest_X-ray_1024_info') 
# El dataframe trae una columna "path" con la ruta y de la imagen y columnas con las clases son: 'No Finding', 'Patology'
# Por lo tanto requiere un preprocesador que lea la columna "path" y convierta las clases en etiquetas binarias.


# Definir el preprocesador y construir los datasets
mobilenet_preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
preprocessor = ImagePreprocessor(
    img_size=IMG_SIZE,
    preprocess_fn=mobilenet_preprocess
)
dataset_builder = DatasetBuilder(
    preprocessor=preprocessor,
    batch_size=32
)
train_ds = dataset_builder.build(train_df, nombres_clases)
test_ds = dataset_builder.build(test_df, nombres_clases)

# Cargar red base preentrenada congelando sus pesos
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# PASO 2: Ensamblar y compilar el modelo
# Ensamblar el modelo clasificador
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(nombres_clases), activation='sigmoid') # Sigmoid para Multi-Label
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(multi_label=True)]
)

# PASO 3: Entrenar el modelo
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds
)


# PASO 4: Guardar los resultados del experimento

# Busca un id al experimento basado en el prefijo y las capetas existentes en resultados
import os

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
model.save(os.path.join(experiment_path, 'mobilenet_xray_model.keras'))

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
eval_results = model.evaluate(test_ds)
with open(os.path.join(experiment_path, 'evaluation_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    for name, value in zip(model.metrics_names, eval_results):
        writer.writerow([name, value])


