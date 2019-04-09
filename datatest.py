from PIL import Image
import os
import cv2
import math as m
import numpy as np
#os.chdir('/media/torbjoern/03C796544677EF72/database/test2')
os.chdir('/home/torbjoern/Pictures/norandom/bypass-r')
print(os.getcwd())

def im_crop():
    for f in os.listdir():
        im = Image.open(f)
        print(im.size)
        print(im.getbbox())
        im2=im.crop(im.getbbox())
        w, h = im2.size
        a_r = round(w/h,1)
        print(a_r)
        im2.save(str(a_r)+f)

def pixelwise(masked, cand):
    #sliding window algorithm
    error = 0
    w, h = masked.size
    print(w,h)
    
    print(cand)
    im = Image.open(cand)
    im.show()
    wc, hc = im.size
    print(wc,hc)
    hyp = m.sqrt(w**2 + h**2)
    hypc = m.sqrt(wc**2 + hc**2)
    print(hyp,hypc)
    rz_masked = masked.resize((w,h),Image.ANTIALIAS)
    rz_masked.show()
        
    #big_picture = np.zeros(wc+20,hc+20)
    #big_picture[10:(10+wc),10:(10+hc)] = cand
    #big_picture.show()
    
    for xw in w:
        for yh in h:
            p_error = im[xw,yh] - cand[xw,yh]
            error += p_error
    return error

def pose_matcher(test):
    candidates = []
    candidates2 = []
    p_error = []
    w, h = test.size
    a_r = round(w/h,1)
    print(a_r)
    for img in os.listdir():
        if img[:3] == str(a_r):
            candidates.append(img)
        elif img[:3] == str(a_r + 0.1) or img[:1] == str(a_r - 0.1):
            candidates2.append(img)
    for img in candidates:
        error = pixelwise(test,img)
        p_error.append(error)
    print(candidates)
    return p_error
image = Image.open('../masked_renault.png')

error = pose_matcher(image)

print(error)

