
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img/fotka.png")
print(img.shape)
blue = img[:,:,0]
green = img[:,:,1]
red = img[:,:,2]
new_red = red.copy()
new_red[(new_red<50) & (new_red>30)] = 0
img[:,:,2] = new_red

plt.plot(np.histogram(blue,bins=256)[0], color="blue", label="blue")
plt.plot(np.histogram(new_red,bins=256, range= (0,255))[0],
         color="red", label="red")
plt.plot(np.histogram(green,bins=256 )[0], color="green", label="green")
plt.legend()
plt.show()

# plt.hist(blue,bins = 256)
# plt.show()

cv2.imshow("jez", img)
cv2.imshow("jez_blue", blue)

img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = img_2[:,:,0]
saturation = img_2[:,:,1]
value = img_2[:,:,2]

cv2.imshow("jez_hue", hue)
cv2.imshow("jez_sat", saturation)
cv2.imshow("jez_v", value)




cv2.waitKey(0)
cv2.destroyAllWindows()