

import img

if __name__ == "__main__":
    # Ejemplo de uso de la función experiment_supres_basico
    params = {
        'dataset_path': '../ds/NIH_Chest_X-ray_1024',
        'dataset_count': 3,        
    }

    images = img.images_load(params['dataset_path'], max_images=params['dataset_count'], verbose=True, recursive=True)
    for image in images:
        img.image_show(image)
    
    
       