import cv2
import torch
import numpy as np
from tracker import *
from tqdm import tqdm

# Khai bao mau
color_xe_dap = (94,200,254)     #cam
color_xe_may = (254,226,102)    #xanh ngoc
color_xe_bus = (102,236,254)    #vang
color_xe_oto = (0,243,104)      #la cay
color_xe_tai = (9, 9, 237)       #do
color_roi    = (184, 19, 193)   #tim

# Khai bao toa do
toado_1 = (840, 400)
toado_2 = (440, 400)
toado_3 = (320, 600)
toado_4 = (970, 600)
toado_check_1 = (350, 550)
toado_check_2 =  (930, 550)
area1 = [toado_1, toado_2, toado_3, toado_4]

#khai bao bien
count = 0  
xe_dap = []
xe_bus = []
xe_oto = []
xe_may = []
xe_tai = []
tracker = Tracker()


# Hàm vẽ khung
def draw_rectangle(frame, x1, y1, x2, y2, color, name):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
    cv2.putText(frame, f"{name} ID:{id} ", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
    cv2.circle(frame, point, 1, color, thickness=2 )

# hàm check và đếm
def check_count(frame, toado, check_1, check_2, id, vehicle_list ):
    if (toado[1] >= check_1[1] and toado[0] >= check_1[0] and toado[0] <= check_2[0]):
        if id not in vehicle_list:
            vehicle_list.append(id)

    

#load model nhận diện
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) 





#load video
cap=cv2.VideoCapture('test.mp4')


# def POINTS(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)
        

# cv2.namedWindow('FRAME')
# cv2.setMouseCallback('FRAME', POINTS)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    
    frame=cv2.resize(frame,(1280,800))
    results=model(frame)
    
    list = []

    #vẽ khu vực nhận diện
    cv2.polylines(frame, [np.array(area1, np.int32)], True, color_roi , 2)
    #đường roi để đếm đối tượng
    cv2.line(frame, toado_check_1, toado_check_2, (0,255,192), thickness=2)

    #lấy các giá trị sau khi nhận diện 
    for index, rows in results.pandas().xyxy[0].iterrows():
        x1 = int(rows['xmin'])
        y1 = int(rows['ymin'])
        x2 = int(rows['xmax'])
        y2 = int(rows['ymax'])
        b = str(rows['name'])
        clas = rows['class']
        list.append([x1, y1, x2, y2, clas])
    idx_bbox = tracker.update(list)

    # lấy lại giá trị sau khi thực hiện track đối tượng và xử lý
    for bbox in idx_bbox:
        x3, y3, x4, y4, id, clas = bbox
        x_midpoint = int((x3+x4)/2)
        y_midpoint = int((y3+y4)/2)
        point = (x_midpoint, y_midpoint)
        if cv2.pointPolygonTest(np.array(area1, np.int32), point, False) >= 0:
            if clas == 0: #Xe_dap
                name = "xe_dap"
                color = color_xe_dap
                draw_rectangle(frame, x3, y3, x4, y4, color, name)
                check_count(frame, point, toado_check_1, toado_check_2, id, xe_dap)

            if clas == 1: #xe_bus
                name = "xe_bus"
                color = color_xe_bus
                draw_rectangle(frame, x3, y3, x4, y4, color, name)
                check_count(frame, point, toado_check_1, toado_check_2, id, xe_bus)

            if clas == 2: #oto
                name = "xe_oto"
                color = color_xe_oto
                draw_rectangle(frame, x3, y3, x4, y4, color, name)
                check_count(frame, point, toado_check_1, toado_check_2, id, xe_oto)
                    
            if clas == 3: #xe_may
                name = "xe_may"
                color = color_xe_may
                draw_rectangle(frame, x3, y3, x4, y4, color, name)
                check_count(frame, point, toado_check_1, toado_check_2, id, xe_may)

            if clas == 4: #xe_tai
                name = "xe_tai"
                color = color_xe_tai
                draw_rectangle(frame, x3, y3, x4, y4, color, name)
                check_count(frame, point, toado_check_1, toado_check_2, id, xe_tai)
        else:
            pass
        
        cv2.putText(frame, f"Xe_oto :{len(xe_oto)} ", (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color_xe_oto, thickness=2)
        cv2.putText(frame, f"Xe_may :{len(xe_may)} ", (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color_xe_may, thickness=2)
        cv2.putText(frame, f"Xe_dap :{len(xe_dap)} ", (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color_xe_dap, thickness=2)
        cv2.putText(frame, f"Xe_bus :{len(xe_bus)} ", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color_xe_bus, thickness=2)
        cv2.putText(frame, f"Xe_tai :{len(xe_tai)} ", (0, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, color_xe_tai, thickness=2)
        total = len(xe_oto) + len(xe_may) + len(xe_dap) +len(xe_bus) + len(xe_tai)
        cv2.putText(frame, f"Tong_xe :{total} ", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_roi, thickness=2)




    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1)&0xFF==27:
        break

    


cap.release()
cv2.destroyAllWindows()
