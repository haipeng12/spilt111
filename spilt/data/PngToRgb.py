from PIL import Image
import cv2 as cv
import os
"This code shows the conversion of images in the dataset to jpg forma"

def PNG_JPG(PngPath):
    img = cv.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)
Path= 'image'
img_dir = os.listdir(Path)
for floder in img_dir:
    ooo=os.path.join(Path, floder)
    files = os.listdir(ooo)
    for img in files:
        print(img)
        if img.endswith('.png'):
            print("mmm")
            PngPath= os.path.join(Path,floder,img)
            PNG_JPG(PngPath)
print("finish!")
