# Guardar el historial de entrenamiento en csv
import csv
import os


def history_save (history, experiment_path, filename = 'training_history.csv'):
    with open(os.path.join(experiment_path, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        # Escribir los encabezados
        headers = list(history.history.keys())
        writer.writerow(headers)
        # Escribir los valores por cada epoch
        for i in range(len(history.history[headers[0]])):
            row = [history.history[h][i] for h in headers]
            writer.writerow(row)
            
            
def evaluation_save (model, test_ds, experiment_path, filename = 'evaluation_results.csv'):
    # Guardar los resultados de la evaluación en csv
    eval_results = model.evaluate(test_ds)
    with open(os.path.join(experiment_path, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for name, value in zip(model.metrics_names, eval_results):
            writer.writerow([name, value])
