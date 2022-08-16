import cv2
import numpy as np
import mediapipe as mp
import math

width = 1920
height = 1080

'''MediaPipe'''
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

std_ratio = 1
t_ratio = 1
ratio = 1

'''초기값'''
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width/3))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height/3))

empty_canvas1 = np.zeros((height,width,3),np.uint8)+255
empty_canvas2 = np.zeros((height,width,3),np.uint8)

icons_name = ['camera.png','draw.png','back.png','resize.png','contrast.png','brightness.png','magic.png']
icons = [cv2.resize(cv2.imread('images/'+name),(100,100)) for name in icons_name]
white = [255,255,255]
black = [0,0,0]
color = [0,0,0]
draw_history = [[]]
floodfill_history = []
work_history = []
capture = np.zeros((1,1,3),np.uint8)+255
captured = None
nowmenu = "draw.png"
ismenu = ""
color_name = "color3"
isdraw = False
isResize = False
isOK = False
isMagic = False
brightness = 0
contrast = 0
throttle = 0
throttle_y = int((1020-340)/2)
menus_position = [] #모든 메뉴의 위치를 저장하기 위함(클릭 시 어떤 메뉴인지 식별)

'''------------------------------------------------------------'''

'''메뉴 레이아웃 구성'''
layout = np.zeros((height,230,3),np.uint8)+100

# 색상 선택 아이콘
for idx in range(3):
    x = [120, 220]
    y = [100 * idx + 10 * idx+10, 100 * idx + 100 + 10 * idx+10]

    color = [0, 0, 0]
    color[idx] = 255
    cv2.rectangle(layout, (x[0], y[0]), (x[1], y[1]), color, -1)
    menus_position.append([x, y, f'color{idx}'])

# 작업 아이콘
for idx, icon in enumerate(icons):
    x = [10, 110]
    y = [100 * idx + 10 * idx+10, 100 * idx + 100 + 10 * idx+10]

    layout[y[0]:y[1], x[0]:x[1]] = icon
    cv2.rectangle(layout, (x[0], y[0]), (x[1], y[1]), [255, 0, 0])
    menus_position.append([x, y, icons_name[idx]])

# save메뉴 따로 제작
cv2.rectangle(layout, (10, 780), (110, 1070), white, -1)
cv2.rectangle(layout, (10, 780), (110, 1070), [255, 0, 0])
cv2.putText(layout, "save", (15, 920), 2, 1.2, black, 2)
menus_position.append([(10, 110), (780, 1070), "save"])

# 스로틀 메뉴 따로 제작

cv2.rectangle(layout,(120,340),(220,1070),white,-1)#범위
menus_position.append([(120,220), (340,1070), "range"])
'''------------------------------------------------------------'''

def cal_ratio(x1,y1,x2,y2):
    x = x2-x1
    y = y2-y1
    return math.sqrt(x ** 2 + y ** 2) * 100

def cal_range(min,max,now):
    k = max - min
    r = 1030 - 340
    temp = int(r/k)
    result = -int((now-340)/temp)
    result-=min
    return result

'''모션 이벤트'''
def motion_event(x, y):
    global canvas, color, image, ismenu
    global color_name,isdraw,draw_history,floodfill_history,isResize,isMagic
    global capture,throttle_y,throttle,brightness,contrast,captured

    for pos in menus_position:
        if(x >= pos[0][0] and x <= pos[0][1]):
            if(y >= pos[1][0] and y <= pos[1][1]):
                if(pos[2] == "range"):
                    throttle_y = y
                else:
                    ismenu = pos[2]
    '''--------------------------'''
    if (ismenu == 'camera.png'):
        while cap.isOpened():
            success, img = cap.read()

            if not success:
                continue
            else:
                draw_history = [[]]
                floodfill_history = []
                capture = img
                captured = capture
                return 0

    if (ismenu[0:5] == "color"):
        color_name = ismenu
        ismenu = "draw.png"
        color = layout[y,x].tolist()

    if (ismenu == "back.png"):
        try:
            if(work_history[-1] == "draw"):
                del draw_history[-1]
                del work_history[-1]
                if(len(draw_history)==0):
                    draw_history = [[]]
        except:
            draw_history = [[]]
        try:
            if(work_history[-1] == "fill"):
                del floodfill_history[-1]
                del work_history[-1]
        except:
            floodfill_history = []

    if (ismenu == "draw.png"):
        isdraw = True
    else:
        isdraw = False

    if (ismenu == "resize.png"):
        isResize = True
    else:
        isResize = False

    if (ismenu == "save"):
        cv2.imwrite("save.png",canvas[:,230:])

    if(x > 230):
        if (ismenu == "draw.png"):
            if(draw_history[-1] != [] and isOK == False):
                draw_history.append([])
                work_history.append("draw")
        if(isdraw):
            draw_history[-1].append([(x,y),color])
        if(isResize):
            floodfill_history = []
            captured = cv2.resize(capture, (x - 230, y))

    if (ismenu == "brightness.png"):
        brightness = cal_range(-60,60,throttle_y)
        throttle = brightness

    if (ismenu == "contrast.png"):
        contrast = cal_range(-30,30,throttle_y)
        throttle = contrast

    if(ismenu == "magic.png"):
        isMagic = True
    else:
        isMagic = False

    if(isMagic):
        if(230 < x < canvas.shape[0] and 0 < y < canvas.shape[1]):
            floodfill_history.append([(x,y),color])
            work_history.append("fill")

def cal_contrast(img,contrast):
    dst = img.copy()
    dst = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    gmin = np.min(img)
    gmax = np.max(img)
    dst = np.clip((img - gmin) * 255. / (gmax - gmin), 0, 255).astype(np.uint8)
    dst = np.clip((1 + contrast/30) * dst - 128 * contrast/30, 0, 255).astype(np.uint8)
    return dst

def cal_brightness(dst,light):
    b,g,r = cv2.split(dst)
    if(light != 0):
        b = cv2.add(b,light)
        g = cv2.add(g, light)
        r = cv2.add(r, light)

    return cv2.merge([b,g,r])

global canvas

empty_canvas1[0:height, 0:230] = layout
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        hand_canvas = empty_canvas2.copy()
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        """--------------------------------------------------------------------"""
        canvas = empty_canvas1.copy()
        try:
            my_img = captured.copy()
            if(my_img.shape[0] > 100):
                my_img = cal_contrast(my_img, contrast)
                my_img = cal_brightness(my_img, brightness)
                canvas[0:my_img.shape[0], 230:230 + my_img.shape[1]] = my_img
        except:
            pass

        for line in draw_history:
            past_point = None
            for l in line:
                if(past_point == None):
                    past_point = l[0]
                else:
                    cv2.line(canvas,past_point,l[0],l[1],3,cv2.LINE_AA)
                    past_point = l[0]

        mask = np.zeros((canvas.shape[0] + 2, canvas.shape[1] + 2), np.uint8)
        mask[:, :230] = 1
        for fill in floodfill_history:
            cv2.floodFill(canvas,mask,fill[0],fill[1],(10,10,10),(10,10,10))

        for pos in menus_position:
            if (pos[2] == ismenu or pos[2] == color_name):
                cv2.rectangle(canvas, (pos[0][0], pos[1][0]), (pos[0][1], pos[1][1]), [255, 255, 0], 2)


        cv2.rectangle(canvas, (120, throttle_y), (220, throttle_y+50), [100, 0, 100], -1)  # 스로틀
        cv2.putText(canvas, f"{throttle}", (130, throttle_y+40), 2, 1.2, white, 2)

        """--------------------------------------------------------------------"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:


                std_ratio = cal_ratio(hand_landmarks.landmark[0].x,hand_landmarks.landmark[0].y,
                                  hand_landmarks.landmark[5].x,hand_landmarks.landmark[5].y)

                t_ratio = cal_ratio(hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y,
                                  hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y)

                ratio = std_ratio / t_ratio

                x = int(hand_landmarks.landmark[4].x*width)
                y = int(hand_landmarks.landmark[4].y*height)



                mp_drawing.draw_landmarks(
                    hand_canvas,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


                if(ratio > 3):
                    cv2.circle(canvas,((width-x),y),30,[255,255,0],-1)
                    motion_event((width-x),y)
                    isOK = True

                else:
                    isOK = False
        hand_canvas = cv2.flip(hand_canvas,1)
        canvas[hand_canvas >0] = hand_canvas[hand_canvas>0]

        cv2.imshow("canvas", canvas)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()