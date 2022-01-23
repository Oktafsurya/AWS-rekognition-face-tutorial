import numpy as np 
import cv2
import boto3
import datetime
import time
import uuid
import pytz
from pytz import timezone


def convert_ts(ts, used_timezone):
    '''Converts a timestamp to the configured timezone. Returns a localized datetime object.'''
    #lambda_tz = timezone('US/Pacific')
    tz = timezone(used_timezone)
    utc = pytz.utc
    
    utc_dt = utc.localize(datetime.datetime.utcfromtimestamp(ts))

    localized_dt = utc_dt.astimezone(tz)

    return localized_dt

scale_factor = .15
green = (0,255,0)
red = (0,0,255)
frame_thickness = 2
cap = cv2.VideoCapture('/dev/video1')
rekognition = boto3.client('rekognition')
COLLECT_NAME: str = 'ProfFaces'

s3_client = boto3.client('s3')
s3_bucket = "face-recognition-video-frame"
s3_key_frames_root = "frames/"

sns_client = boto3.client('sns')
label_watch_phone_num = "+6282136292661"

writer = None
allowed_name = ['okta', 'ariel', 'tzuyu']

while(True):

    detected_valid_name = []

    # Capture frame-by-frame
    ret, frame = cap.read()
    H_ORI, W_ORI, channels = frame.shape

    # Convert frame to jpg
    # small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
    ret, buf = cv2.imencode('.jpg', frame)

    faces_detect = rekognition.detect_faces(Image={'Bytes':buf.tobytes()}, Attributes=['ALL'])

    faces = rekognition.search_faces_by_image(CollectionId=COLLECT_NAME,
                                           Image={'Bytes': buf.tobytes()}, MaxFaces=10)

    faces = faces['FaceMatches']
    faces_detect = faces_detect['FaceDetails']

    if len(faces) == 0:
        faces.extend(['UNKNOWN'])

    if len(faces_detect) == 0:
        faces_detect.extend(['UNKNOWN'])

    print('[INFO] len faces recognition result:', len(faces))
    print('[INFO] faces recognition result:')
    print(faces)
    print(f'*'*60)
    print('[INFO] len faces detection result:', len(faces_detect))
    print('[INFO] faces detection result:')
    print(faces_detect)
    print(f'='*60)

    assert len(faces) == len(faces_detect)

    used_timezone = "Asia/Jakarta"
    frame_id = str(uuid.uuid4())
    now_ts = time.time()
    now = convert_ts(now_ts, used_timezone)
    year = now.strftime("%Y")

    mon = now.strftime("%m")
    day = now.strftime("%d")
    hour = now.strftime("%H")

    # Draw rectangle around faces
    # face = faces['FaceMatches']
    names_list = []
    similarity = None
    if len(faces) > 0:
        for idx in range(len(faces)):
            if faces[idx] != 'UNKNOWN':
                similarity = faces[idx]['Similarity']
                name = str(faces[idx]['Face']['ExternalImageId']).split('.')[0]
                text_name = f'Name: {name} - {similarity}%'
                # x1 = int(faces[idx]['Face']['BoundingBox']['Left']*W_ORI)
                # y1 =  int(faces[idx]['Face']['BoundingBox']['Top']*H_ORI)
                # x2 = int((faces[idx]['Face']['BoundingBox']['Left']+faces[idx]['Face']['BoundingBox']['Width'])*W_ORI)
                # y2 = int((faces[idx]['Face']['BoundingBox']['Top']+faces[idx]['Face']['BoundingBox']['Height'])*H_ORI)
            else: 
                similarity = 'UNKNOWN'
                name = 'UNKNOWN'
                text_name = f'Name: Unknown'
                # x1 = 200
                # y1 = 0
                # x2 = 300
                # y2 = 100

            names_list.append(name)

            if faces_detect[idx] != 'UNKNOWN':
                smile = faces_detect[idx]['Smile']['Value']
                gender_label = faces_detect[idx]['Gender']['Value']
                gender_score = faces_detect[idx]['Gender']['Confidence']
                age_low = faces_detect[idx]['AgeRange']['Low']
                age_high = faces_detect[idx]['AgeRange']['High']
                emotion_type = faces_detect[idx]['Emotions'][0]['Type']
                emotion_score = faces_detect[idx]['Emotions'][0]['Confidence']
                text_gender = f'{gender_label} - {gender_score}%'
                text_age = f'interval age: {age_low} - {age_high}'
                text_emotion = f'Emotion: {emotion_type} - {emotion_score}%'
                x1 = int(faces_detect[idx]['BoundingBox']['Left']*W_ORI)
                y1 =  int(faces_detect[idx]['BoundingBox']['Top']*H_ORI)
                x2 = int((faces_detect[idx]['BoundingBox']['Left']+faces_detect[idx]['BoundingBox']['Width'])*W_ORI)
                y2 = int((faces_detect[idx]['BoundingBox']['Top']+faces_detect[idx]['BoundingBox']['Height'])*H_ORI)
            else:
                text_gender = 'Unknown'
                text_age = f'interval age: Unknown'
                text_emotion = f'Emotion: Unknown'
                smile = False

            # cv2.rectangle(frame, (x1,y1), (x2,y2), green if smile else red, frame_thickness)
            cv2.putText(frame, text_gender, (x1,y1), 1, 1, (0,0,255), 2)
            cv2.putText(frame, text_age, (x1,y1-30), 1, 1, (0,0,255), 2)
            cv2.putText(frame, text_emotion, (x1,y2), 1, 1, (0,0,255), 2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), green if name != 'UNKNOWN'  else red, frame_thickness)
            cv2.putText(frame, text_name, (x1,y2-30), 2, 1, (0,0,255), 2)
            # cv2.putText(frame, text_name, (x1,y2), 2, 1, (0,0,255), 2)
    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('out.avi', fourcc, 30, (W_ORI, H_ORI), True)

    writer.write(frame)

    for name in names_list:
        if name in allowed_name:
            detected_valid_name.append(name)

    #Send out notification(s), if needdetected_nameed
    if len(detected_valid_name) > 0 and label_watch_phone_num:
        notification_txt = 'On {}...\n'.format(now.strftime('%x, %-I:%M %p %Z'))

        for name in detected_valid_name:
            notification_txt += '- "{}" was detected with {}% confidence.\n'.format(
                name,
                similarity)

        print(notification_txt)

        if label_watch_phone_num:
            sns_client.publish(PhoneNumber=label_watch_phone_num, Message=notification_txt)

    s3_key = (s3_key_frames_root + '{}/{}/{}/{}/{}.jpg').format(year, mon, day, hour, frame_id)
    ret, buf = cv2.imencode('.jpg', frame)
    s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=buf.tobytes()
    )
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
writer.release()
cv2.destroyAllWindows()

# orang menghadap camera, klik tombol untuk recognize