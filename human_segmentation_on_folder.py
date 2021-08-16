from tensorflow import keras
import os
from glob import glob 
import cv2
import numpy as np

from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool

def timeout(seconds):
    def decorator(function):
        def wrapper(*args, **kwargs):
            pool = ThreadPool(processes=1)
            result = pool.apply_async(function, args=args, kwds=kwargs)
            try:
                return result.get(timeout=seconds)
            except TimeoutError as e:
                return e
        return wrapper
    return decorator

@timeout(6)
def get_ch():
    return int(input("1. for my model \n2. for his model:"))

ch = get_ch()
print(ch)
exit
if isinstance(ch, TimeoutError):
    print('\n\nYou took too long so we will go with default 1. model')
    ch=1
else:
    print('Okay !')
# ch=int(input("1. for my model \n2. for his model:"))

if ch==1:
    model=keras.models.load_model("C:/Users/home/Downloads/deep (4).h5",compile=False) #loss: 0.0358 - dice_score: 0.9642 - val_loss: 0.0364 - val_dice_score: 0.9636
elif ch==2:
    model=keras.models.load_model("C:/Users/home/Downloads/model.h5",compile=False) #his model
        
if not os.path.exists("C:/Users/home/Desktop/segmentation/test/"):
    os.mkdir("C:/Users/home/Desktop/segmentation/test/")

allpath=glob("C:/Users/home/Desktop/segmentation/test/*")

if not os.path.exists("C:/Users/home/Desktop/segmentation/results/"):
    os.mkdir("C:/Users/home/Desktop/segmentation/results/")


for i,path in enumerate(allpath):
    input=cv2.imread(path,cv2.IMREAD_COLOR)
    processed_input=cv2.resize(input,(512,512))
    processed_input=np.expand_dims(processed_input,axis=0)/255
    out=model.predict(processed_input)
    out=(out>0.5)*255
    out=np.squeeze(out,axis=0).astype(np.uint8)
    # out=np.repeat(out,3,axis=-1)
    # out=np.squeeze(out,axis=-1)
    # print(out.shape,input.shape[0],input.shape[1])
    out=cv2.resize(out,(input.shape[1],input.shape[0]))
    out=np.expand_dims(out,axis=-1)
    out=np.concatenate([input,out],axis=-1).astype(np.uint8)
    print(out.shape)
    cv2.imwrite(f"C:/Users/home/Desktop/segmentation/results/{i}.png",out)


