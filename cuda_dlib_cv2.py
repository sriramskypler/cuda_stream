import dlib
print('dlib version:', dlib.__version__)
print('CUDA support:', dlib.DLIB_USE_CUDA)
import cv2
print('OpenCV version:', cv2.__version__)
print('CUDA support:', cv2.cuda.getCudaEnabledDeviceCount())
