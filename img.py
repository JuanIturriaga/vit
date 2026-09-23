import random

import cv2 as cv
import numpy as np
import os

def image_load(path, mode='rgb', metadata=False):
    #cargar una imagen desde un archivo
    map = {
        'rgb': cv.IMREAD_COLOR,
        'bgr': cv.IMREAD_COLOR,
        'grayscale': cv.IMREAD_GRAYSCALE,
        'unchanged': cv.IMREAD_UNCHANGED
    }
    img = cv.imread(path, map.get(mode,  map.get(mode)))
    if mode == 'rgb' and img is not None:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
    if metadata:
        #obtener información del archivo, path, filename, extension, size, channels, tamaño en bytes
        filename, file_extension = os.path.splitext(os.path.basename(path))
        file_size = os.path.getsize(path)
        height, width = img.shape[:2] if img is not None else (None, None)
        channels = img.shape[2] if img is not None and len(img.shape) == 3 else 1
        metadata_dict = {
            'path': path,
            'filename': filename,
            'extension': file_extension,
            'width': width,
            'height': height,
            'channels': channels,
            'bytes': file_size, 
            'mode': mode
        }
        return img, metadata_dict
    else:
        return img

def image_save(path, image, mode='rgb'):
    #guardar una imagen en un archivo
    if mode == 'rgb':
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(path, image)
    
def images_load(folder_path, mode='rgb', extensions=['.png', '.jpg', '.jpeg'], recursive=False, offset=0, max_images=None, dim=(-1,-1), verbose=False, metadata=False):
    #cargar todas las imagenes de una carpeta
    
    images = []
    metadata_list = []
    i = 0;
    if recursive:
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if any(filename.endswith(ext) for ext in extensions) :
                    if i >= offset:
                        if metadata:
                            img, meta = image_load(os.path.join(root, filename), mode, metadata=True)
                            meta['index'] = i                        
                        else:
                            img = image_load(os.path.join(root, filename), mode)
                        if dim != (-1,-1):
                            img = image_resize(img, dim[0], dim[1], interpolation='bicubic')
                        if img is not None:
                            images.append(img)
                            if metadata:
                                metadata_list.append(meta)
                            if max_images is not None and len(images) >= max_images:
                                break
                    i += 1
            if max_images is not None and len(images) >= max_images:
                break
    else:
        for filename in os.listdir(folder_path):
            if any(filename.endswith(ext) for ext in extensions):
                if i >= offset:
                    if metadata:
                        img, meta = image_load(os.path.join(folder_path, filename), mode, metadata=True)
                        meta['index'] = i
                    else:
                        img = image_load(os.path.join(folder_path, filename), mode)
                    if dim != (-1,-1):
                        img = image_resize(img, dim[0], dim[1], interpolation='bicubic')
                    if img is not None:
                        images.append(img)
                        if metadata:
                            metadata_list.append(meta)
                        if max_images is not None and len(images) >= max_images:
                            break
                i += 1
    if verbose:
        print(f"Cargadas {len(images)} imágenes desde {folder_path}")
        
    if metadata:
        return np.array(images), metadata_list
    else:
        return np.array(images)

def images_save (folder_path, images, mode='rgb', prefix='image_', start_index=0):
    #guardar una lista de imagenes (numpy array) en una carpeta
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, img in enumerate(images):
        image_save(os.path.join(folder_path, f"{prefix}{start_index + i:06d}.png"), img, mode)

def image_resize(image, width, height, interpolation='bilinear'):
    #redimensionar una imagen a un tamaño específico
    map = {
        'bilinear': cv.INTER_LINEAR,
        'nearest': cv.INTER_NEAREST,
        'bicubic': cv.INTER_CUBIC,
        'area': cv.INTER_AREA,
        'lanc': cv.INTER_LANCZOS4
    }
    interpolation_value = map.get(interpolation)    
    if interpolation_value is None:
        raise ValueError(f"Invalid interpolation method: {interpolation}")
    return cv.resize(image, (width, height), interpolation=interpolation_value)

def images_resize(images, width, height, interpolation='bilinear'):
    #redimensionar una lista de imagenes a un tamaño específico
    return np.array([image_resize(image, width, height, interpolation) for image in images])

def image_blur(image, kernel_size=5):
    #aplicar un blur gaussiano a una imagen
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

def images_blur(images, kernel_size=5):
    #aplicar un blur gaussiano a una lista de imagenes
    return np.array([image_blur(image, kernel_size) for image in images])

def image_map_compare (img_orginal, img_test, method='diff'):
    #compara dos imagenes y devuelve una imagen con la diferencia
    if method == 'diff':
        return cv.absdiff(img_orginal, img_test)
    elif method == 'sub':
        return cv.subtract(img_orginal, img_test)
    elif method == 'add':
        return cv.add(img_orginal, img_test)
    else:
        raise ValueError(f"Invalid comparison method: {method}")

def image_unsharp_mask(image, kernel_size=5, amount=1.0):
    #aplicar un unsharp mask a una imagen
    blurred = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    sharpened = cv.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened
    
    
def image_show (image, title="Image", color='rgb', wait_key=0):
    #mostrar una imagen en una ventana
    
    if color == 'rgb':
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    cv.imshow(title, image)
    cv.waitKey(wait_key)
    cv.destroyAllWindows()

def test_unsharp_mask():
    image_file = ".\\results\\xray_home_000009\\images_xray_home_000009\\predicts\\image_000000.png"
    image_output = ".\\results\\xray_home_000009\\images_xray_home_000009\\predicts\\image_000000_sharpened.png"
    image = image_load(image_file, mode="rgb")
    sharpened_image = image_unsharp_mask(image, kernel_size=5, amount=1.0)
    image_show(image, title="Original Image", color='rgb', wait_key=0)
    image_show(sharpened_image, title="Sharpened Image", color='rgb', wait_key=0)
    image_save(image_output, sharpened_image, mode="rgb")


    


def test_generacion_images():
    
    images, meta_data = images_load(".\\ds\\random_shapes_64", recursive=True, max_images=6,  mode='rgb', metadata=True)
    
    #for image in images:
    #    if image is not None:
    #        print(f"Original Image Shape: {image.shape}")
    #        resized_image = image_resize(image, 256, 256, interpolation='bicubic')
    #        #image_show(resized_image, title="Resized Image", color='rgb', wait_key=0)
    #    else:
    #        print("Error: No se pudo cargar la imagen.")
    
    
    # hacer un shuffle de las imágenes y guardar el ínice en el que quedó cada una en los metadatos
    
    seed = random.randint(0, 100000)  # Semilla aleatoria para reproducibilidad
    
    rng = np.random.default_rng(seed=seed)  # Semilla opcional para reproducibilidad
    indices = rng.permutation(len(images))
    
    images = images[indices]
    
    for i in range(len(meta_data)):
        meta_data[i]['shuffled_index'] = indices[i]
        
    # crear dataframe con la metadata de las imágenes
    import pandas as pd
    df_metadata = pd.DataFrame(meta_data)
    print(df_metadata)
    
    # con mat plot lib mostrar las 6 imágnes con su indice actual y el indice original
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(f"Shuffled Index: {i}\nOriginal Index: {df_metadata.iloc[i]['shuffled_index']}")
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()
    
    
    
if __name__ == "__main__":
    test_unsharp_mask()
