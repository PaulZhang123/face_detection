#!/usr/bin/python3

import threading
import cv2
import time
from queue import Queue
import os
import configparser

'''
1、opencv api
2、multi-thread
'''

#GIL全局锁
# 单核只能并发，多核才能并行

def job(l,q):
    for i in range(len(l)):
        l[i]=l[i]**2 #幂
    # return l
    q.put(l)

def multithreading():
    q = Queue()
    threads = []
    data = [[1,2,3],[3,4,5],[4,4,4],[5,5,5]]
    for i in range(4):
        t = threading.Thread(target=job,args=(data[i],q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    results = []
    for _ in range(4):
        results.append(q.get())
    print(results)

def thread_job():
    # print("This is an added Thread,number is %s" % threading.current_thread())
    print("T1 start\n")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finish\n")

def main():
    added_thread = threading.Thread(target=thread_job,name='T1')
    added_thread.start()
    # print(threading.active_count())
    # print(threading.enumerate())
    # print(threading.current_thread())
    added_thread.join()
    print('all done\n')

def job1():
    global A,lock
    lock.acquire()
    for i in range(10):
        A+=1
        print('job1',A)
    lock.release()

def job2():
    global A,lock
    lock.acquire()
    for i in range(10):
        A+=10
        print('job2',A)
    lock.release()


def draw_rectangle(ret):
    pass

def fr(img_path):
    for root,dirs,files in os.walk(img_path):
        for file in files:
            if file.endswith('mp4'):
                print(root)
                print(file)


def fd(img_path,result_path):
    print("1")
    print(img_path)
    # 级联分类器
    detect_face =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for root,dirs,files in os.walk(img_path):
        for file in files:
            # print(file)
            if file.endswith('JPG'):
                pic = os.path.join(root,file)
                cap = cv2.imread(pic)
                gray = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
                # detect = detect_face.detectMultiScale(gray,scaleFacto=1.2,minNeighbors=15)
                detect = detect_face.detectMultiScale(gray,1.2,2,minSize=(100,100))
                print(detect)
                for l,t,w,h in detect:
                    cv2.rectangle(cap,(l,t),(l+w,t+h),(255,180,0),10)
                    # draw_rectangle(x)
                path_out_img = os.path.join(result_path,'img')
                if not os.path.exists(path_out_img):
                    os.makedirs(path_out_img)
                img_save = os.path.join(path_out_img,file)
                cv2.imwrite(img_save,cap)

            elif file.endswith('mp4'):
                video = os.path.join(root,file)
                cap = cv2.VideoCapture(video)
                #视频属性
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #获取原视频的宽
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #获取原视频的搞
                fps = int(cap.get(cv2.CAP_PROP_FPS)) #帧率
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) #视频的编码

                path_out_video = os.path.join(result_path,'video')
                if not os.path.exists(path_out_video):
                    os.makedirs(path_out_video)
                video_save = os.path.join(path_out_video,file)

                #视频对象的输出
                out = cv2.VideoWriter(video_save, fourcc, fps, (width, height))
                while cap.isOpened():
                    ret,frame = cap.read()
                    cv2.imshow('frame',frame)
                    key = cv2.waitKey(25)
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    detect = detect_face.detectMultiScale(gray,1.2,2,minSize=(100,100))
                    for l,t,w,h in detect:
                        cv2.rectangle(frame,(l,t),(l+w,t+h),(255,180,0),10)
                    out.write(frame)
                cap.release()
                out.release()
                cv2.destroyAllWindows()
      
def statistics():
    # 获取当前的时间戳
    start = time.perf_counter()
    print("1")
    time.sleep(1)
    end = time.perf_counter()
    runtime = end-start
    print("运行时间:",runtime)


def cpu():
    pass

def mem():
    pass

# 线程定时触发
def createTimer():
    t = threading.Timer(3, repeat)
    t.start()

def repeat():
    createTimer()
    print('Now-1:', time.strftime('%H:%M:%S',time.localtime()))
    # time.sleep(3)
    # print('Now-2:', time.strftime('%H:%M:%S',time.localtime()))
# createTimer()

if __name__=='__main__':
    config =  configparser.ConfigParser()
    config.read('./cfg.ini')
    img_path = config['cfg']['img_path']
    result_path = config['cfg']['result_path']

    print(img_path,result_path)
    fd(img_path,result_path)
    
    # main()
    # multithreading()

    # # Lock
    # lock = threading.Lock()
    # A=0
    # t1 = threading.Thread(target=job1)
    # t2 = threading.Thread(target=job2)
    # t1.start()
    # t2.start()
    # t1.join()
    # t2.join()

    
    # 如果一个线程正在做一件比较耗时的事，能够中断线程，等待，重启线程继续工作? 线程定时器

    # t1 = threading.Thread(target=fr,args=(pic,video,img_path,result_path))
    # t1.start()
    # t2 = threading.Thread(target=statistics)
    # t2.start()

    # nt1 = threading.Thread(target=cpu,args=(1,),name='nt1')
    # nt2 = threading.Thread(target=mem,args=(1,),name='nt2')
    # nt1.start()
    # nt2.start()
    # print(threading.enumerate())


    
    
