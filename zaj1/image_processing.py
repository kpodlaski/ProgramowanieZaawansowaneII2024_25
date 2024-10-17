
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img/fotka.png")
print(img.shape)
blue = img[:,:,0]
green = img[:,:,1]
red = img[:,:,2]
new_red = red.copy()
new_red[new_red>200] = 0

plt.plot(np.histogram(blue,bins=256)[0], color="blue", label="blue")
plt.plot(np.histogram(new_red,bins=256, range= (0,255))[0],
         color="red", label="red")
plt.plot(np.histogram(green,bins=256 )[0], color="green", label="green")
plt.legend()
plt.show()

cv2.imshow("jez", img)
cv2.imshow("jez_green", blue)

cv2.waitKey(0)
cv2.destroyAllWindows()