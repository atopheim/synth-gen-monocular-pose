from PIL import Image
import os

os.chdir('/media/torbjoern/03C796544677EF72/database/test2')
print(os.getcwd())

def im_crop()
    for f in os.listdir():
        im = Image.open(f)
        print(im.size)
        print(im.getbbox())
        im2=im.crop(im.getbbox())
        w, h = im2.size
        a_r = round(w/h,1)
        print(a_r)
        im2.save(str(a_r)+f)

def pixelwise(masked, cand)
    #sliding window algorithm
    error = 0
    w, h = masked.size
    wc, hc = cand.size
    hyp = sqrt(w**2 + h**2)
    hypc = sqrt(wc**2 + hc**2)
    if hyp < hypc:
        rz_masked = cv2.resize(masked,None,fx=hypc/hyp,fy=hypc/hyp)
    
    big_picture = np.zeros(wc+20,hc+20)
    big_picture[10:(10+wc),10:(10+hc)] = cand
    cv2.imshow(big_picture)
    cv2.waitKey(1000)
    
    for xw in w:
        for yh in h:
            p_error = test[xw,yh] - cand[xw,yh]
            error += p_error
    return error

def pose_matcher(test) 
    candidates = []
    candidates2 = []
    p_error = []
    a_r = round(w/h,1)
    for img in os.listdir():
        if img[:2] == a_r:
            candidates.append(img)
        elif img[:2] == (a_r + 0.1) or img[:1] == (a_r - 0.1):
            candidates2.append(img)
    for img in candidates:
        p_error.append(pixelwise(test,img))

    

