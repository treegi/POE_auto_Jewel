
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.image as mpimg
import time
import numpy as np
import copy
import time
import numpy as np
from PIL import ImageGrab


define_step = ["改造石","增幅石","富豪石","蛻變石","重鑄石","種類讀取錯誤","無偵測到珠寶","完成"]
times_about_step =["補充光環+珠寶洞", "果斷神諭+珠寶洞", "果斷神諭", "補充光環"]
times_about ={
    "補充光環+珠寶洞" : 0,
    "果斷神諭+珠寶洞" : 0,
    "果斷神諭" : 0,
    "補充光環" : 0
}

class J_type:
    def __init__(self, define_step):
        self.reset(define_step)
    def reset(self, define_step, j_show=True):
        self.Presence_bool = False
        self.Harbinger_bool = False
        self.Socket_bool = False
        self.Check = 0
        self.NumberOfAffix = 0
        self.define_step = define_step
        self.times = 0
        self.kind_of_bool = 0
        self.j_show = j_show
    def set_data(self,Presence_bool, Harbinger_bool, Socket_bool, NumberOfAffix, Check):
        self.reset(self.define_step)
        self.Presence_bool = Presence_bool
        self.Harbinger_bool = Harbinger_bool
        self.Socket_bool = Socket_bool
        self.NumberOfAffix = NumberOfAffix
        self.Check = Check
        if Presence_bool == True:
            self.kind_of_bool += 1
        if Harbinger_bool == True:
            self.kind_of_bool += 1
        if Socket_bool == True:
            self.kind_of_bool += 1
    def show_kind(self,step):
        if (self.j_show == True):
            if ((self.Check == 0) and (self.NumberOfAffix ==0)):
                print("無偵測到珠寶")
                return
            print("偵測結果", end = ": ")
            if (self.Presence_bool):
                print("補充光環", end = ", ")
            if (self.Harbinger_bool):
                print("果斷神諭", end = ", ")
            if (self.Socket_bool):
                print("珠寶洞", end = ", ")
            if (self.kind_of_bool == 0):
                print("都沒有", end = ", ")
            print("下一步驟為->" + define_step[step])
    def times_about_cal(self):
        if ((self.Presence_bool == 1) and (self.Socket_bool == 1) ):
            times_about[times_about_step[0]] += 1
        elif ((self.Harbinger_bool == 1) and (self.Socket_bool == 1) ):
            times_about[times_about_step[1]] += 1
        elif ((self.Harbinger_bool == 1)):
            times_about[times_about_step[2]] += 1
        elif ((self.Presence_bool == 1)):
            times_about[times_about_step[3]] += 1
        else:
            None
    def times_about_cal_show(self):
        print("半成品數量統計:",end="")
        for key in times_about_step:
            print(key + ":" + str(times_about[key]), end=" ,")
        print("")
    
    def next_step(self):
        if ((self.Check > 0) or (self.NumberOfAffix >0)):
            self.times_about_cal_show()
            if (self.NumberOfAffix == 0):
                self.show_kind(define_step.index("蛻變石"))
                return define_step.index("蛻變石")

            elif (self.NumberOfAffix == 1):
                if ((self.kind_of_bool == 1)):
                    self.show_kind(define_step.index("增幅石"))
                    return define_step.index("增幅石")
                if ((self.kind_of_bool == 0)):
                    self.show_kind(define_step.index("改造石"))
                    return define_step.index("改造石")
                else:
                    print("NumberOfAffix = 1例外")
                    return -1

            elif (self.NumberOfAffix == 2):
                if ((self.kind_of_bool == 2) or (self.Harbinger_bool == 1) or (self.Presence_bool == 1)):#有果斷或補充就富豪
                    self.show_kind(define_step.index("富豪石"))
                    return define_step.index("富豪石")
                elif ((self.kind_of_bool <= 1)):#都沒中
                    self.show_kind(define_step.index("改造石"))
                    return define_step.index("改造石")
                else:
                    print("NumberOfAffix = 2例外")
                    return -1

            elif (self.NumberOfAffix >= 3):
                if ((self.kind_of_bool <= 2)):
                    if ((self.Presence_bool == True) and (self.Harbinger_bool == True) ):
                        self.show_kind(define_step.index("完成"))
                        return define_step.index("完成")
                    else:
                        self.show_kind(define_step.index("重鑄石"))
                        self.times_about_cal()
                        return define_step.index("重鑄石")#記得要recheck
                elif ((self.kind_of_bool == 3)):
                    self.show_kind(define_step.index("完成"))
                    return define_step.index("完成")
                else:
                    print("NumberOfAffix > 3例外")
                    return -1
            else:
                self.show_kind(define_step.index("種類讀取錯誤"))
                return define_step.index("種類讀取錯誤")
        else:
            self.show_kind(define_step.index("無偵測到珠寶"))
            return define_step.index("無偵測到珠寶")


class Auto_jewel:
    def __init__(self, define_step, show = False):
        self.Presence = cv2.cvtColor(cv2.imread(os.getcwd()+'\\source\\data\\Presence.png'), cv2.COLOR_RGB2GRAY)
        self.Harbinger = cv2.cvtColor(cv2.imread(os.getcwd()+'\\source\\data\\Harbinger.png'), cv2.COLOR_RGB2GRAY)
        self.Socket = cv2.cvtColor(cv2.imread(os.getcwd()+'\\source\\data\\Socket.png'), cv2.COLOR_RGB2GRAY)
        self.NumberOfAffix = cv2.cvtColor(cv2.imread(os.getcwd()+'\\source\\data\\NumberOfAffix.png'), cv2.COLOR_RGB2GRAY)
        self.Check = cv2.cvtColor(cv2.imread(os.getcwd()+'\\source\\data\\Check.png'), cv2.COLOR_RGB2GRAY)
        self.threshold = 0.8
        self.show = show
        self.j_type = J_type(define_step)
    def get_screen(self):
        return np.array(ImageGrab.grab((0, 0, 760, 1015)))
    def match(self, img, goal, threshold):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #grayscale conversion
        ret,thresh1 = cv2.threshold(gray,130,255,cv2.THRESH_BINARY)
        res = cv2.matchTemplate(thresh1, goal, cv2.TM_CCOEFF_NORMED) #设定阈值 threshold = 0.7 #res大于70% loc = np.where( res >= threshold)
        loc = np.where( res >= threshold)
        """
        if (self.show):
            new_img = copy.deepcopy(img)
            w, h = goal.shape
            for pt in zip(*loc[::-1]): 
                cv2.rectangle(new_img, pt, (pt[0] + h, pt[1] + w), (0, 255, 0), 2)
            plt.imshow(thresh1, cmap='gray')
            plt.show()
        """
        return len(loc[0])
    def predict(self):
        img = self.get_screen()
        p = self.match(img, self.Presence, self.threshold)
        h = self.match(img, self.Harbinger, self.threshold)
        s = self.match(img, self.Socket, self.threshold)
        num = self.match(img, self.NumberOfAffix, self.threshold)
        check = self.match(img, self.Check, self.threshold)
        self.j_type.set_data(Presence_bool = p, Harbinger_bool = h, Socket_bool = s, NumberOfAffix = num, Check = check)
        ans1 = self.j_type.next_step()
        return ans1


    



        




"""

now=time.time()

image = cv2.imread('test.png')
img_gray = cv2.imread('test2.png')

gray_text = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion
ret_text,thresh1_text = cv2.threshold(gray_text,130,255,cv2.THRESH_BINARY)

gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY) #grayscale conversion
ret,thresh1 = cv2.threshold(gray,130,255,cv2.THRESH_BINARY)

Presence = thresh1_text[130:150,50:80]
Harbinger = thresh1_text[110:130,120:380]
Socket = thresh1_text[150:170,120:370]

cv2.imwrite('times.png', Presence)
cv2.imwrite('Harbinger.png', Harbinger)
cv2.imwrite('Socket.png', Socket)

threshold = 0.8

res_Harbinger = cv2.matchTemplate(thresh1,Harbinger,cv2.TM_CCOEFF_NORMED) #设定阈值 threshold = 0.7 #res大于70% loc = np.where( res >= threshold)
loc_Harbinger = np.where( res_Harbinger >= threshold)

res_Presence = cv2.matchTemplate(thresh1,Presence,cv2.TM_CCOEFF_NORMED) #设定阈值 threshold = 0.7 #res大于70% loc = np.where( res >= threshold)
loc_Presence = np.where( res_Presence >= threshold)

res_Socket = cv2.matchTemplate(thresh1,Socket,cv2.TM_CCOEFF_NORMED) #设定阈值 threshold = 0.7 #res大于70% loc = np.where( res >= threshold)
loc_Socket = np.where( res_Socket >= threshold)

w, h = Harbinger.shape
for pt in zip(*loc_Harbinger[::-1]): 
    cv2.rectangle(img_gray, pt, (pt[0] + h, pt[1] + w), (0, 255, 0), 2)

print(time.time()-now)



gray_text = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #grayscale conversion
ret_text,thresh1= cv2.threshold(gray_text,130,255,cv2.THRESH_BINARY)





"""

"""
cv2.imshow('Detected',img_gray)

plt.imshow(Harbinger, cmap='gray')
plt.show()

Presence = thresh1_text[25:50,65:210]
plt.imshow(Presence, cmap='gray')
plt.show()

cv2.imwrite('Check.png', Presence)

plt.imshow(a, cmap='gray')
plt.show()


plt.imshow(thresh1_text, cmap='gray')
plt.show()

plt.imshow(gray, cmap='gray')
plt.show()

plt.imshow(thresh1[283:360,323:360], cmap='gray')
plt.show()

"""
"""
import pytesseract
pytesseract.pytesseract.tesseract_cmd ="C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = 'C:\\Program Files (x86)\\Tesseract-OCR\\tessdata'
text = pytesseract.image_to_string(thresh1, lang='chi_tra')
"""