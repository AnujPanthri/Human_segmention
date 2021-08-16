# import plaidml.keras
# plaidml.keras.install_backend()
import tensorflow as tf
# from tensorflow import keras
import keras
# from os import environ
# environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ['KERAS_BACKEND'] = "keras.backend"

import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)
# model=keras.models.load_model("C:/Users/home/Downloads/deep (4).h5",compile=False) #loss: 0.0358 - dice_score: 0.9642 - val_loss: 0.0364 - val_dice_score: 0.9636
# model=keras.models.load_model("C:/Users/home/Downloads/deep (3).h5",compile=False) #96.05 val dice score
# model=keras.models.load_model("C:/Users/home/Downloads/deep (2).h5",compile=False)
# model=keras.models.load_model("C:/Users/home/Downloads/deep (1).h5",compile=False)
# model=keras.models.load_model("C:/Users/home/Downloads/model.h5",compile=False) #his model
# model=keras.models.load_model("C:/Users/home/Downloads/deep.h5",compile=False)
# model=keras.models.load_model("C:/Users/home/Downloads/deep_dice.h5",compile=False)
# model.summary()

def main():
    # model=unet()
    # model.load_weights("C:/Users/home/Downloads/deep_dice.h5")
    # unetmodel=keras.models.load_model("C:/Users/home/Downloads/deep_dice.h5",compile=False)
    # hismodel=keras.models.load_model("C:/Users/home/Downloads/model.h5",compile=False)
    model=keras.models.load_model("C:/Users/home/Downloads/deep (4).h5",compile=False)
    vid = cv2.VideoCapture(0)
    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0
    while(True):
        new_frame_time = time.time()
 
        # Calculating the fps
    
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
    
        print("fps:",fps)
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        frame=cv2.resize(frame,(512, 512))
        cv2.imshow('frame', frame)
        output=getout(model,frame)
        cv2.imshow('my model(deeplab)', output)
        # output=getout(unetmodel,frame)
        # cv2.imshow('my model(unet)', output)
        # output=getout(hismodel,frame)
        # cv2.imshow('his model', output)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def getout(model,input):
    h,w=input.shape[0],input.shape[1]
    input=cv2.resize(input,(512, 512))
    input=np.expand_dims(input,axis=0)
    input=input/255
    # print(input.shape)

    out=model.predict(input)
    out=(out>0.5)*1
    # print(out.shape)
    final=input*out
    final=final.squeeze(axis=0)
    final=final*255
    final=final.astype("uint8")
    # final=cv2.resize(final,(w,h))
    # print(final.shape)
    # plt.imshow(final)
    # plt.show()
    return final


if __name__=="__main__":
    main()