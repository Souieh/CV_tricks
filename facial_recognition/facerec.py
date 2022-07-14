import face_recognition
import cv2 , os
import numpy as np
import pandas as pd



def outline(shapes , dots , dotted=True , closed= True ,resize_factor=1 ):
    '''newdots = [dots[0]]
    for dotindex in range(1,len(dots)-1):
         newdots.append(interpo(dots[dotindex],dots[dotindex-1],dots[dotindex+1] ))
    newdots.append(dots[-1])
    dots = newdots
    print(dots)'''
    if(len(dots)<2):
        return shapes
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
    alpha = 0.5
    bg_color=(100,100,100)
    font_color = (255,255,255)  
    font_scale = 3* resize_factor/16
    font = cv2.FONT_HERSHEY_PLAIN

    for (top, right, bottom, left), info in facial_info:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= resize_factor
        right *= resize_factor
        bottom *= resize_factor
        left *= resize_factor



        if(info==-1):
            fullname = "Uknown"
            gender = "Uknown"
            Nationality = "Uknown"
            age = "Uknown"
        else:
            fullname = "{}.{}".format(fd.loc[info, 'surname'][0].capitalize(),fd.loc[info, 'firstname'].capitalize()).capitalize()
            gender = fd.loc[info, 'gender']
            Nationality = fd.loc[info, 'nationality']
            age="{}".format(fd.loc[info, "age"])

        hls = ["Fullname : "+ fullname,
                "Age : "  + age,
                "Gender : "+ gender.capitalize(),
                "Nationality : " + Nationality.capitalize()]
         

        cv2.rectangle(shapes,(left, top ), (right, bottom ),  bg_color, 1)
        cv2.rectangle(shapes,(left, bottom), (right, bottom + len(hls)*resize_factor*4 + 12), bg_color, cv2.FILLED)
        
        for hl_index,hl in enumerate(hls) :
            cv2.putText(shapes, hl, 
                            (left + 6, bottom + (hl_index+1)*resize_factor*4 + 6), 
                            font, font_scale,
                            font_color, 1)
        

    for marks in facial_marks :
        for facial_mark in marks:
            is_closed = False
            if(facial_mark=="right_eye" or facial_mark=="left_eye"):
                is_closed = True
            shapes=outline(shapes,marks[facial_mark]    ,closed=is_closed ,resize_factor=resize_factor)


    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, 1-alpha, shapes,  alpha, 0)[mask]
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

def main(path):
    known_face_encodings,known_face_indexs , fd =  datafaces(path)
        
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    #path_to = os.path.join(root, path_to)
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    ratio =1/ frame_height* frame_width 

  
    while (True):
        # Grab a single frame of video
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        #higher resolution means higher accuracy but low fps
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # bgr to rgb
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
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    face_index = known_face_indexs[best_match_index]
                    info = face_index
                face_names.append(info)

        process_this_frame = not process_this_frame

        frame = cv2.resize(frame, (0,0) , fx=2,fy=2)
        out = print_Facials(frame, fd,  zip(face_locations, face_names), face_landmarks , resize_factor=8)
        out = cv2.resize(out, (int(720*ratio),720) )
        cv2.imshow("faces", out)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(".\\faces\\" )