from dataset_builder import DatasetBuilder, ImagePreprocessor
from custom_models import create_model_conv_classifier
from experiment_preset import get_experiment_preset
from evaluation import history_save, evaluation_save
from load_data import load_dataframe_xray
import tensorflow as tf
from dataset_builder import DatasetBuilder
import os

# paso 0: obtener la configuración del experimento
params = {
    'img_size': 244,
    'train_batch_size': 32,
    'train_epochs': 200,
    'train_learning_rate': 1e-3,
    'experiment_prefix': 'XRAY_CONV_',
    'results_path': './results',
    'model': 'conv_classifier',    
    'num_classes': -1
}

experiment_path, experiment_id = get_experiment_preset(params)

print(f"Experiment path: {experiment_path}, Experiment ID: {experiment_id}")

# paso 1: construir el dataset
# Cargar y preprocesar los datos
nombres_clases = ['No Finding', 'Patology']
NUM_CLASSES = len(nombres_clases)
params['num_classes'] = NUM_CLASSES

train_df, test_df = load_dataframe_xray(dataset_info_path = './ds/NIH_Chest_X-ray_1024_info') 
# El dataframe trae una columna "path" con la ruta y de la imagen y columnas con las clases son: 'No Finding', 'Patology'
# Por lo tanto requiere un preprocesador que lea la columna "path" y convierta las clases en etiquetas binarias.

IMG_SIZE = params.get('img_size', 244)
BATCH_SIZE = params.get('train_batch_size', 32)

# Definir el preprocesador y construir los datasets

#preprocesdor que normaliza la imágen
preprocess = lambda x: x / 255.0

preprocessor = ImagePreprocessor(
    img_size=(IMG_SIZE, IMG_SIZE),
    preprocess_fn=preprocess
)
dataset_builder = DatasetBuilder(
    preprocessor=preprocessor,
    batch_size=BATCH_SIZE
)
train_ds = dataset_builder.build(train_df, nombres_clases)
test_ds = dataset_builder.build(test_df, nombres_clases)

# paso 2: crear y compilar el modelo
LEARNING_RATE = params.get('train_learning_rate', 1e-3)
LOSS_FUNCTION = params.get('train_loss_function', 'binary_crossentropy')

model = create_model_conv_classifier(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    num_classes=len(nombres_clases)
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=LOSS_FUNCTION,
    metrics=[tf.keras.metrics.AUC(multi_label=True)]
)
model.summary()

# paso 3: entrenar el modelo

EPOCHS = params.get('train_epochs', 200)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

model.save(os.path.join(experiment_path, "model_conv_classifier.keras"))

# paso 4: guardar los resultados
history_save(history, experiment_path)
evaluation_save(model, test_ds, experiment_path)

#save json params
import json
with open(os.path.join(experiment_path, "params.json"), "w") as f:
    json.dump(params, f)

