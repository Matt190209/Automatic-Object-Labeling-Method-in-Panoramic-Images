import numpy as np
import sys, math, os
from PIL import Image
from datetime import datetime
import glob
import imageio.v2 as imageio
from multiprocessing import Process

SIZE = 3072
HSIZE = SIZE / 2.0

def process_side(i, im, output_subdir, base_filename):
    color_side = np.zeros((SIZE, SIZE, 3), np.uint8)
    
    it = np.nditer(np.zeros((SIZE, SIZE), np.uint8), flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        axA = it.multi_index[0]
        axB = it.multi_index[1]
        z = -axA + HSIZE
        
        if i == 0:
            x = HSIZE
            y = -axB + HSIZE
        elif i == 1:
            x = -HSIZE
            y = axB - HSIZE
        elif i == 2:
            x = axB - HSIZE
            y = HSIZE
        elif i == 3:
            x = -axB + HSIZE
            y = -HSIZE
        elif i == 4:
            z = HSIZE
            x = axB - HSIZE
            y = axA - HSIZE
        elif i == 5:
            z = -HSIZE
            x = axB - HSIZE
            y = -axA + HSIZE

        r = math.sqrt(float(x*x + y*y + z*z))
        theta = math.acos(float(z)/r)
        phi = -math.atan2(float(y), x)
        
        ix = int((im.shape[1]-1)*phi/(2*math.pi))
        iy = int((im.shape[0]-1)*(theta)/math.pi)
        
        r = im[iy, ix, 0]
        g = im[iy, ix, 1]
        b = im[iy, ix, 2]
        
        color_side[axA, axB, 0] = r
        color_side[axA, axB, 1] = g
        color_side[axA, axB, 2] = b

        it.iternext()

    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    pimg = Image.fromarray(color_side)
    pimg.save(os.path.join(output_subdir, f"{base_filename}_side_{i}.jpg"), quality=85)

def cubemap(filename, output_dir):
    im = imageio.imread(filename)
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_subdir = os.path.join(output_dir, base_filename)
    processes = []
    for i in range(0, 6):
        p = Process(target=process_side, args=(i, im, output_subdir, base_filename))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    combine_sides(base_filename, output_subdir, output_dir)

def combine_sides(base_filename, input_dir, output_dir):
    # Create an empty image for the cubemap
    cubemap_img = Image.new('RGB', (4 * SIZE, 3 * SIZE))
    
    # Load each side image
    sides = [Image.open(os.path.join(input_dir, f"{base_filename}_side_{i}.jpg")) for i in range(6)]
    
    # Place the images in the correct position
    cubemap_img.paste(sides[0], (SIZE, 0))         # Top
    cubemap_img.paste(sides[1], (SIZE, 2 * SIZE))  # Bottom
    cubemap_img.paste(sides[2], (SIZE, SIZE))      # Front
    cubemap_img.paste(sides[3], (0, SIZE))         # Left
    cubemap_img.paste(sides[4], (2 * SIZE, SIZE))  # Right
    cubemap_img.paste(sides[5], (3 * SIZE, SIZE))  # Back
    
    # Save the combined image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cubemap_img.save(os.path.join(output_dir, f"{base_filename}_cubemap.jpg"), quality=85)

if __name__ == "__main__":
    panorama_image_dir = "D:\\Proyecto Final\\PF\\Aprendizaje_No_Supervisado\\Fast_R-CNN\\Etiquetado-Manual\\360fotosnew"
    output_dir = "D:\\Proyecto Final\\PF\\Aprendizaje_No_Supervisado\\Fast_R-CNN\\Etiquetado-Manual\\sub_pano"
    
    for filename in glob.glob(os.path.join(panorama_image_dir, '*.jpg')):
        startTime = datetime.now()
        cubemap(filename, output_dir)
        print(filename, datetime.now() - startTime)