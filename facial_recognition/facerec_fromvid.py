import face_recognition
import cv2 , os
import numpy as np
import pandas as pd
video_capture = cv2.VideoCapture(0)

def interpo(pt,pt1,pt2,factor=1/4):
    pt0 = pt[0]*(1-2*factor) + (pt2[0] + pt1[0])*factor
    pt1 = pt[1]*(1-2*factor) + (pt2[1] + pt1[1])*factor
    return (pt0,pt1)


def outline(shapes , dots , dotted=True , closed= True ,resize_factor=1 ):
    
    for ord in range(len(dots)-1):
        cv2.line(shapes, (dots[ord][0]*resize_factor,dots[ord][1]*resize_factor)
        , (dots[ord+1][0]*resize_factor,dots[ord+1][1]*resize_factor)
        , (255,255,255))
        if(dotted):
            cv2.circle(shapes,   (dots[ord][0]*resize_factor,dots[ord][1]*resize_factor), 
            4, (255,255,255) , cv2.FILLED)
        pass
    if(closed):
        cv2.line(shapes,  (dots[-1][0]*resize_factor,dots[-1][1]*resize_factor),  
         (dots[0][0]*resize_factor,dots[0][1]*resize_factor) , 
        (255,255,255))
    if(dotted):
        cv2.circle(shapes,   (dots[-1][0]*resize_factor,dots[-1][1]*resize_factor), 
        4, (255,255,255),cv2.FILLED)
    return shapes

def print_Facials(frame ,fd, facial_info,facial_marks ,resize_factor=1):
    out = frame.copy()
    #  facial_info= zip(face_locations, face_names)
    shapes = np.zeros_like(frame, np.uint8)
    alpha = 0.2
    for (top, right, bottom, left), info in facial_info:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= resize_factor
        right *= resize_factor
        bottom *= resize_factor
        left *= resize_factor

        # Draw a box around the face
        cv2.rectangle(shapes,(left, top ), (right, bottom ), (200, 200, 200 ), 1)

        # Draw a label with a name below the face
        cv2.rectangle(shapes,(left, bottom - 35), (right, bottom + 40), (200, 200, 200 ), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX
        if(info==-1):
            fullname = "Uknown"
            gender = "Uknown"
            Nationality = "Uknown"
            RH = "Uknown"
            age = "Uknown"
        else:
            fullname = "{}.{}".format(fd.loc[info, 'surname'][0],fd.loc[info, 'firstname'])
            gender = fd.loc[info, 'gender']
            Nationality = fd.loc[info, 'nationality']
            age="{}".format(fd.loc[info, "age"])
            
        cv2.putText(shapes, "Fullname : "+ fullname, (left + 6, bottom - 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(shapes, "Age : "  + age, (left + 6, bottom - 6 ), font, 0.5, (255, 255, 255), 1)
        cv2.putText(shapes, "Gender : "  + gender, (left + 6, bottom + 8 ), font, 0.5, (255, 255, 255), 1)
        cv2.putText(shapes, "Nationality : " + Nationality, (left + 6, bottom + 22 ), font, 0.5, (255, 255, 255), 1)

    for lmrk in facial_marks :
        shapes=outline(shapes,lmrk['right_eyebrow']   ,closed=False , resize_factor=resize_factor)
        shapes=outline(shapes,lmrk['left_eyebrow']    ,closed=False , resize_factor=resize_factor)
        shapes=outline(shapes,lmrk["right_eye"]       ,resize_factor=resize_factor)
        shapes=outline(shapes,lmrk["left_eye"]        ,resize_factor=resize_factor)
        shapes=outline(shapes,lmrk["top_lip"]         ,resize_factor=resize_factor)
        shapes=outline(shapes,lmrk["bottom_lip"]      ,resize_factor=resize_factor)
    
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
    return out

def datafaces(path):
    try:
        fd = pd.read_csv(os.path.join(path,"faces_data.csv"))
    except ValueError:
        print(ValueError)

    fd = fd.reset_index()
    
    known_face_encodings=[]
    known_face_indexs=[]
    for index, row in fd.iterrows() :
        filepath = os.path.join(path,row['filename'])
        person_img =  face_recognition.load_image_file(filepath)
        known_face_encodings.append(face_recognition.face_encodings(person_img)[0])
        known_face_indexs.append(index)
    return  known_face_encodings,known_face_indexs ,fd

def main(path , fname , path_to="out"):
    known_face_encodings,known_face_indexs , fd =  datafaces(path)
        
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    #path_to = os.path.join(root, path_to)
    cap = cv2.VideoCapture(fname)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    outbuff = cv2.VideoWriter(path_to+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    ijj = 0
    while (cap.isOpened()== True):
        # Grab a single frame of video
        ret, frame = cap.read()
        if ret == True:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = small_frame[:, :, ::-1]
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_landmarks = face_recognition.face_landmarks(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    #     name = known_face_names[first_match_index]
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    info = -1
                    
                    '''
                    if True in matches:
                        first_match_index = matches.index(True)
                        face_index =  known_face_indexs[first_match_index]
                        name = "{}\nCostumer".format(df[df['id']==face_index]['name'])
                    '''
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        face_index = known_face_indexs[best_match_index]
                        info = face_index
                    face_names.append(info)
            process_this_frame = not process_this_frame
            frame = cv2.resize(frame, (0,0) , fx=2,fy=2)
            out = print_Facials(frame, fd,  zip(face_locations, face_names), face_landmarks , resize_factor=4)
            out = cv2.resize(out, (0,0) , fx=0.5,fy=0.5)
            outbuff.write(out)
            
        else: 
            break
        ijj+=1
        os.system("cls")
        print("totale doen : {} frames".format(ijj))
    cap.release()
    outbuff.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(".\\faces\\" ,"test.mp4",path_to="o2")