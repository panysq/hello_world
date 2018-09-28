import cv2


im = cv2.imread('111.bmp')
print(im.shape)
b,g,r = cv2.split(im)
cv2.imshow('b',b)
cv2.imshow('g',g)
cv2.imshow('r',r)
cv2.waitKey(0)


img_show = cv2.imread('mura_an_2.bmp')
entropy_gray = cv2.imread('2_entropy_gray.png')
average_gray = cv2.imread('2_average_gray.png')
print(entropy_gray.dtype)
cv2.namedWindow('entropy_gray', 0)
cv2.imshow("entropy_gray", entropy_gray)
result_1 = cv2.threshold(entropy_gray, 55, 255, cv2.THRESH_OTSU)[1]
result_2 = cv2.threshold(average_gray, 55, 255, cv2.THRESH_OTSU)[1]
cv2.namedWindow('result_1', 0)
cv2.imshow("result_1", result_1)
cv2.namedWindow('result_2', 0)
cv2.imshow("result_2", result_2)
_, contours, hierarchy = cv2.findContours(result_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_show = cv2.drawContours(img_show, contours, -1, (0, 0, 255), 2)
# 画出瑕疵轮廓
_, contours, hierarchy = cv2.findContours(result_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_show = cv2.drawContours(img_show, contours, -1, (0, 255, 0), 2)  # 画出瑕疵轮廓
cv2.namedWindow('img_show_1', 0)
cv2.imshow("img_show_1", img_show)
cv2.imwrite('2_res.png', img_show)
cv2.waitKey(0)
