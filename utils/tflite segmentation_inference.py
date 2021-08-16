import tensorflow as tf
from tensorflow import keras
# import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

print(tf.__version__)
import numpy as np
def get_out(img,idx):
  # Load TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter("C:/Users/home/Downloads/lite-model_deeplabv3_1_metadata_2.tflite")

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.allocate_tensors()

  new_img=img
  new_img=cv2.resize(new_img,(257,257))
  new_img=np.expand_dims(new_img,axis=0).astype('float32')/255
  # print(new_img.shape)
  interpreter.set_tensor(input_details[0]['index'], new_img)
      
  # run the inference
  interpreter.invoke()

  # output_details[0]['index'] = the index which provides the input
  output_data = interpreter.get_tensor(output_details[0]['index'])
  output_data=(output_data>0.5)*1
  output_data=output_data[...,idx:idx+1]
  output_data=new_img*output_data
  output_data=np.squeeze(output_data,axis=0)
  output_data=output_data*255
  output_data=output_data.astype("uint8")
  return output_data

def main():
    # model=unet()
    # model.load_weights("C:/Users/home/Downloads/deep_dice.h5")
    # model=keras.models.load_model("C:/Users/home/Downloads/deep_dice.h5",compile=False)
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
        frame=cv2.resize(frame,(257, 257))
        cv2.imshow('frame', frame)
        # output=get_out(frame,15)
        # cv2.imshow('output15', output)
        # output=get_out(frame,16)
        # cv2.imshow('output16', output)
        output=get_out(frame,17)
        cv2.imshow('output17', output)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()  

