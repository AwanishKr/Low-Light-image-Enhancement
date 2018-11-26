import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA             # to calclate the frobenius norm of matrices


def initialize(image):
	global T_present, G_present, Z_present, u_present
	rows, cols = image.shape
	T_present = np.zeros((rows, cols), np.float64)
	G_present = np.zeros((2*rows, cols), np.float64)
	Z_present = np.zeros((2*rows, cols), np.float64)
	u_present = 2

def update_T():
    global T_hat, u_present, D_matrix, G_present, Z_present, D_h, D_v, T_updated
    num = np.fft.fft2(2*T_hat + u_present*np.matmul(D_matrix.transpose(), (G_present-Z_present/u_present)))
    den = 2 + u_present*(np.conjugate(np.fft.fft2(D_h))*np.fft.fft2(D_h) + np.conjugate(np.fft.fft2(D_v))*np.fft.fft2(D_v))
    T_updated = np.abs(np.fft.ifft2(num/den))
    
def update_G():
	global T_updated, grad_T_updated, Z_present, u_present, G_updated, Weight_matrix
	alpha = 10
	value1 = cv2.Sobel(T_updated, cv2.CV_64F, 1, 0, ksize=5)
	value2 = cv2.Sobel(T_updated, cv2.CV_64F, 1, 0, ksize=5)
	grad_T_updated = np.concatenate((value1, value2), axis=0)
	value = grad_T_updated + (Z_present/u_present)
	G_updated = np.zeros((grad_T_updated.shape[0], grad_T_updated.shape[1]), np.float64)

	Weight_matrix = np.ones((2*rows, cols), np.float64)

	A_tmp = (alpha/u_present)*Weight_matrix
	for i in range(A_tmp.shape[0]):
		for j in range(A_tmp.shape[1]):
			G_updated[i,j] = np.sign(value[i,j])*max((np.abs(value[i,j])-A_tmp[i,j]), 0)

def update_Z():
	global Z_updated, Z_present, u_present, grad_T_updated, G_updated
	Z_updated  = Z_present + u_present*(grad_T_updated - G_updated)

def update_u():
	global u_present, u_updated
	prow = 8
	u_updated = u_present*prow


if __name__ == '__main__':
    global T_hat,T_present,T_updated,G_present,G_updated,Z_present,Z_updated,u_present,u_updated,grad_T_present,grad_T_updated,D_matrix, D_h, D_v
    delta = 0.00001

    capt_img = cv2.imread("lamp.png", 1)
    capt_img = cv2.resize(capt_img, (capt_img.shape[0], capt_img.shape[0], ))
    print("shape of input image", capt_img.shape)
    rows, cols, channels = capt_img.shape
    
    rec_img = np.zeros((rows, cols, channels), np.float64)
    T_hat = np.zeros((rows, cols), np.float64)
    for i in range(0, rows):
        for j in range(0, cols):
            T_hat[i,j] = np.max(capt_img[i,j])
    
    T_hat = T_hat + 0.0002
    print("shape of T_hat", T_hat.shape)

    I_matrix = np.identity(rows, dtype = float)

    D_h = cv2.Sobel(I_matrix, cv2.CV_64F, 1, 0, ksize=5)
    D_v = cv2.Sobel(I_matrix, cv2.CV_64F, 0, 1, ksize=5)
    D_matrix = np.concatenate((D_h, D_v), axis=0)
    print("shape of D_matrix", D_matrix.shape)
    initialize(T_hat)
    update_u()
    update_T()
    update_G()
    update_Z()
    number = 0
    while(LA.norm(grad_T_updated - G_updated) > delta*LA.norm(T_hat)):
    	print("still in the loop, optimizing")
    	update_u()
    	update_T()
    	update_G()
    	update_Z()
    	number = number+1
    	if number == 100:
    		break
    # reconstruction
    print("number of iterations used", number)
    
    rec_img[:,:,0] = capt_img[:,:,0]/T_updated
    rec_img[:,:,1] = capt_img[:,:,1]/T_updated
    rec_img[:,:,2] = capt_img[:,:,2]/T_updated
    
    max_value = np.max(rec_img)
    print("maximum value is: ", max_value)

    # for i in range(rec_img.shape[0]):
    # 	for j in range(rec_img.shape[1]):
    # 		for k in range(3):
    # 			if rec_img[i,j,k] > 0.9*max_value:
    # 				rec_img[i,j,k] = rec_img[i,j,k]*0.5

    rec_img = cv2.GaussianBlur(rec_img,(3,3),0)
    cv2.imshow("recovered image", rec_img)
    cv2.imshow("captured image", capt_img)
    cv2.waitKey(0)