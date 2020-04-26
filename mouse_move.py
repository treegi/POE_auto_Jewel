import pyautogui
import random
import time
import operator
import sys
import os
import playsound 
pyautogui.PAUSE = 0.2
pyautogui.FAILSAFE = True      # 啟用自動防故障功能
width,height = pyautogui.size()   # 螢幕的寬度和高度
pyautogui.position()        # 滑鼠當前位置

from detect import *
a = Auto_jewel(define_step)

position_dict = {
    "中間" : [[290,339],[410,530]],
    "蛻變石" : [[44,69],[286,312]],
    "改造石" : [[98,118],[287,307]],
    "富豪石" : [[394,415],[285,309]],
    "增幅石" : [[206,231],[335,361]],
    "重鑄石" : [[152,177],[453,482]],
}
def get_screen_pixel(pixel):
    return pyautogui.screenshot().getpixel(pixel)

def mouse_right_click(next):
    ans_before = get_screen_pixel(pyautogui.position())
    pyautogui.click(button='right')
    ans_after = get_screen_pixel(pyautogui.position())
    if (operator.eq(ans_before,ans_after)):
        print(define_step[next]+"已用完，中止")
        play_voice(1)
        sys.exit()
        return 0
    else:
        return 1


def mouse_move_point(square, duration=0.25):
    x = int(random.random()*(square[0][1]-square[0][0])+square[0][0])
    y = int(random.random()*(square[1][1]-square[1][0])+square[1][0])
    pyautogui.moveTo(x,y,duration=duration)
    time.sleep(random.random()*0.3+0.1)
    check = pyautogui.position() 
    if ((x == int(check[0])) and (y == int(check[1]))): 
        return 1
    else:
        print("使用者控制滑鼠，終止")
        return 0

cost_dict = {
    "蛻變石" : 0,
    "改造石" : 0,
    "富豪石" : 0,
    "增幅石" : 0,
    "重鑄石" : 0,
}

def cost_cal(key):
    cost_dict[key]+=1
    print("花費統計:",end="")
    for key in cost_dict:
        print(key + ":" + str(cost_dict[key]), end=" ,")
    print("")

def play_voice(mode):
    file_success = os.getcwd()+'\\source\\audio\\laugh.mp3'
    not_enouth_stone_success = os.getcwd()+'\\source\\audio\\stone.mp3'
    if (mode == 0):
        playsound.playsound(file_success)
    else:
        playsound.playsound(not_enouth_stone_success)


while(True):
    time.sleep(0.2)
    next = a.predict()
    if ((next <= 4) and (next >= 0)):
        ans = mouse_move_point(position_dict[define_step[next]], duration=0.25)
        if (ans == 0):
            play_voice(1)
            break

        ans = mouse_right_click(next)
        if (ans == 0):
            play_voice(1)
            break

        ans = mouse_move_point(position_dict["中間"], duration=0.25)
        if (ans == 0):
            play_voice(1)
            break
        pyautogui.click(button='left')
        cost_cal(define_step[next])
    elif (define_step[next] == "完成"):
        print("成功")
        play_voice(0)
        break
    elif (define_step[next] == "無偵測到珠寶"):
        ans = mouse_move_point(position_dict["中間"], duration=0.25)
        if (ans == 0):
            play_voice(1)
            break
    else:
        print("非預期錯誤")
        play_voice(1)
        break







