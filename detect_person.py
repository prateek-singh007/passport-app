import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation

face_position = []


# extra function begin here
def contour_detect(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    roi_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        height = im.shape[1]
        if h < (height * 0.4):
            continue
        roi_contour.append(contour)
    return roi_contour


def blur_image(ima, x):
    image = ima.copy()
    blurred_img = cv2.GaussianBlur(image, (21, 21), 0)
    mask = np.zeros(image.shape, np.uint8)

    cv2.drawContours(mask, x, -1, (255, 255, 255), 5)
    output = np.where(mask == np.array([255, 255, 255]), blurred_img, image)
    return output


'''def roi_person(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ma = 0
    roi_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        height = im.shape[1]
        if h < (height * 0.4):
            continue
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        ik = im.copy()
        ik = cv2.rectangle(ik, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #cv2.imshow('im', ik)
        #cv2.waitKey(0)
    #final_image = im[y1:(y1 + h1), x1:(x1 + w1)]
    return im'''
# extra function ends here


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def reduce_size(image, parameter, option):
    size = (parameter[0], parameter[1])
    ik2 = image_resize(image, size[0], size[1])
    return ik2


def remove(image, parameter):
    global face_position
    segmentor = SelfiSegmentation()
    white = (255, 255, 255)
    imgNoBg = segmentor.removeBG(image, white, threshold=.7)

    #print(imgNoBg.shape)
    face_position[0] = face_position[0]-(int(0.3*face_position[2]))-parameter[4]
    face_position[1] = face_position[1]-(int(0.4*face_position[3]))-parameter[2]
    face_position[2] = face_position[2]+(int(0.6*face_position[2]))+parameter[5]
    face_position[3] = face_position[3]+(int(0.8*face_position[3]))+parameter[3]

    parameter[3]=0
    if face_position[0]<0:
        parameter[4] = 0 - face_position[0]
        face_position[0]=0
    if face_position[1]<0:
        parameter [2] = 0-face_position[1]
        face_position[1]=0
    if face_position[2]>imgNoBg.shape[1]:
        parameter[5] = face_position[2]-imgNoBg.shape[1]
        face_position[2]=imgNoBg.shape[1]
    if face_position[3]>imgNoBg.shape[0]:
        parameter[3] = 0
        face_position[3]=imgNoBg.shape[0]

    #print(*face_position)
    imgNoBg = imgNoBg[face_position[1]:face_position[1]+face_position[3],face_position[0]:face_position[0]+face_position[2]]
    imgNoBg = cv2.copyMakeBorder(imgNoBg, parameter[2], parameter[3], parameter[4], parameter[5], cv2.BORDER_CONSTANT,
                            value=(255, 255, 255))

    '''# show both images
    # cv2.imshow('office', image)
    cv2.imshow('office no bg', imgNoBg)
    cv2.waitKey(0)

    # roi coordinate
    xx=parameter[4]
    yy=parameter[2]
    hh = imgNoBg.shape[0]-parameter[3]
    ww = imgNoBg.shape[1]-parameter[5]
    print(imgNoBg.shape)
    imgNoBg = imgNoBg[yy:hh,xx:ww]'''
    #cv2.imwrite('iiii.jpg',imgNoBg)
    # end
    imgNoBg = reduce_size(imgNoBg, parameter, 1)
    """cv2.imshow('img',imgNoBg)
    cv2.waitKey(0)"""
    x = contour_detect(imgNoBg)
    xx = blur_image(imgNoBg, x)
    # xx = cv2.copyMakeBorder(xx, parameter[2], parameter[3], parameter[4], parameter[5], cv2.BORDER_CONSTANT,value=(255, 255, 255))
    xx = reduce_size(xx, parameter, 0)

    # reduce_size(xx)
    return xx


def detect(img2):
    flag = 0
    global face_position
    # image = cv2.resize(image, (640, 480))
    # image=img2.copy()
    image = img2.copy()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    face = []
    for c in faces:
        if (c[2] * c[3]) > 14000:
            face.append(c)

    if len(face) >= 1:
        for (x, y, w, h) in faces:
            # cv2.circle(image, (x + (w // 2), y + (h // 2)), 5, (255, 0, 0), 30)
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                flag = 1
                face_position = [x,y,w,h]
            '''for (ex, ey, ew, eh) in eyes:
                pass
                # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 3)'''
            # cv2.imshow('img', image)
    #print(image.shape)
    return flag


def solve(image, size):
    flag = detect(image)
    if flag == 1:
        imge = remove(image, size)
        return imge
    else:
        return []


if __name__ == '__main__':
    img = cv2.imread('i7.jpg')
    size = (480, 480)
    solve(img, size)
    cv2.destroyAllWindows()
