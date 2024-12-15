from PIL import Image, ImageFile
from os import path
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def rot(img_path, angle):
    image = Image.open(img_path)
    rot_img_path = img_path.replace('@map@', f'@rot{angle}map@')

    # Rotate the image by 90 degrees
    rotated_image = image.rotate(angle, expand=True)

    # Save the rotated image
    rotated_image.save(rot_img_path)

basedir = r'D:\.cache\QDRaw'
basedir = r'F:\.cache\QDRaw'

map_dirs = {
    "2012": path.join(basedir, '201209', '@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),
    "2013": path.join(basedir, '201310', '@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),  
    "2017": path.join(basedir, '201710', '@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),
    "2019": path.join(basedir, '201911', '@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),
    "2020": path.join(basedir, '202002', '@map@120.421142578125@36.6064453125@120.48418521881104@36.573829650878906@.jpg'),  
    "2022": path.join(basedir, '202202', '@map@120.42118549346924@36.60643328438966@120.4841423034668@36.573836401969416@.jpg'), 
}

for year, map_dir in map_dirs.items():
    rot(map_dir, 90)