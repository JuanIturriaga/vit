import os


def get_experiment_preset(params):


    RESULTS_PATH = params.get('results_path', './results')
    EXPERIMENT_PREFIX = params.get('experiment_prefix', 'XRAY_VIT_')


    existing_experiments = [d for d in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, d))]
    experiment_ids = [int(d.replace(EXPERIMENT_PREFIX, '')) for d in existing_experiments if d.startswith(EXPERIMENT_PREFIX) and d.replace(EXPERIMENT_PREFIX, '').isdigit()]
    next_experiment_id = max(experiment_ids, default=0) + 1
    experiment_id = f"{EXPERIMENT_PREFIX}{next_experiment_id:04}"
    experiment_path = os.path.join(RESULTS_PATH, experiment_id)
    os.makedirs(experiment_path, exist_ok=True)
    
    params['experiment_path'] = experiment_path
    params['experiment_id'] = experiment_id
    
    return experiment_path, experiment_id