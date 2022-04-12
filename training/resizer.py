from PIL import Image
import os, sys

path = r"C:/Users\ASUS/Desktop/cleanlab mangoes/images/test/"
dirs = os.listdir( path )

def resize():
    # print('oye',dirs)
    for item in dirs:
        
        # print(item)
        im = Image.open(path+item)
        f, e = os.path.splitext(path+item)
        # im.thumbnail((256,256), Image.ANTIALIAS)
        imResize = im.resize((256,256), Image.ANTIALIAS)
        imResize.save(f + '.jpg', 'JPEG', quality=100)

resize()