import numpy as np 
import face_collections as fcol
from pathlib import Path
from pprint import pprint
import cv2
import boto3

FACE_SEARCH_DIR = 'faces_to_match'
COLLECT_NAME: str = 'ProfFaces'
green = (0,255,0)
red = (0,0,255)
frame_thickness = 2

img_fname = str(Path(FACE_SEARCH_DIR) / 'tzuyu.png')
input_img = cv2.imread(img_fname)
H_ORI, W_ORI, channels = input_img.shape

print('Searching collection for', img_fname)

_, im_arr = cv2.imencode('.jpg', input_img)  
im_bytes = im_arr.tobytes()
rekognition = boto3.client('rekognition')
faces = rekognition.detect_faces(Image={'Bytes':im_bytes})
face = faces['FaceDetails'][0]

# try to find the face in the collection
faces_info = fcol.find_face(COLLECT_NAME,
                            img_fname)

print('Found', len(faces_info),
      'match' + ('' if len(faces_info) == 1 else 's'))

# Extract the name of the reference image(s) that were matched
pprint([face_info['Face']['ExternalImageId'] for face_info in faces_info])

if len(faces_info) > 0:
      similiraty = faces_info[0]['Similarity']
      # x1 = int(faces_info[0]['Face']['BoundingBox']['Left']*W_ORI)
      # y1 = int(faces_info[0]['Face']['BoundingBox']['Top']*H_ORI)
      # x2 = int((faces_info[0]['Face']['BoundingBox']['Left'] + faces_info[0]['Face']['BoundingBox']['Width'])*W_ORI)
      # y2 = int((faces_info[0]['Face']['BoundingBox']['Top'] + faces_info[0]['Face']['BoundingBox']['Height'])*H_ORI)
      x1 = int(face['BoundingBox']['Left']*W_ORI)
      y1 =  int(face['BoundingBox']['Top']*H_ORI)
      x2 = int((face['BoundingBox']['Left']+face['BoundingBox']['Width'])*W_ORI)
      y2 = int((face['BoundingBox']['Top']+face['BoundingBox']['Height'])*H_ORI)
      name = str(faces_info[0]['Face']['ExternalImageId']).split('.')[0]
      text = f'{name} - {similiraty}%'
      cv2.rectangle(input_img, (x1,y1), (x2,y2), green, frame_thickness)
      cv2.putText(input_img, text, (x1,y1), 2, 1, (0,0,255), 2)
else:
      name = 'UNKNOWN'
      cv2.putText(input_img, name, (100,100), 2, 1, red, 2)

input_img = cv2.resize(input_img, (900, 900))
cv2.imshow('frame', input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()