import os
from PIL import Image

def merge_images(folder_path, output_filename):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_files.sort(key=lambda x: int(x.split('_')[0]))
    
    images = []
    for f in image_files:
        img_path = os.path.join(folder_path, f)
        img = Image.open(img_path)
        images.append(img)
    
    img_width, img_height = images[0].size
    
    gap = 10
    
    cols = 5
    rows = (len(images) + cols - 1) // cols
    
    merged_width = (img_width + gap) * cols - gap
    merged_height = (img_height + gap) * rows - gap
    
    merged_image = Image.new('RGB', (merged_width, merged_height), (255, 255, 255))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * (img_width + gap)
        y = row * (img_height + gap)
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        merged_image.paste(img, (x, y))
    
    merged_image.save(output_filename)
    print(f"Saved: {output_filename}")

if __name__ == "__main__":
    result_folder = os.path.join(os.path.dirname(__file__), 'result')
    merge_images(result_folder, 'merged_result.png')
    
    primitive_folder = os.path.join(os.path.dirname(__file__), 'result_primitivation')
    merge_images(primitive_folder, 'merged_primitive.png')
