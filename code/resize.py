import os
import glob

original_path = "./alpha_small/"

files = os.listdir(original_path)
print(files)

raise RuntimeError

fromat = ".png"

for (path, files) in os.walk(original_path):
    sp = path[:-(len(path)-len(original_path))]
    for file in files:
        if file.endswith(tuple(format)):
            image = Image.open(path+"/"+file)
            image = image.resize(28,28)

            image.save("./alphabet_dataset"+sp+ +"/"+file)
