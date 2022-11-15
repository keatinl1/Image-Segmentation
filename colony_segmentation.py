import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

path = './'
images = sorted(glob.glob(f'{path}/*.jpeg'))

for image in images:
    nemo_std = cv2.imread(image)

    nemo = cv2.cvtColor(nemo_std, cv2.COLOR_BGR2HSV)

    light_orange = (25,20,200)
    dark_orange = (35, 80, 230)

    mask = cv2.inRange(nemo, light_orange, dark_orange)

    result = cv2.bitwise_and(nemo, nemo, mask=mask)

    final = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

    height, width, depth = final.shape

    height, width, depth = nemo.shape
    center_coordinates = (int(width/2), int(height/2))
    radius = 570
    color = (0, 0, 0)
    thickness = 500

    kernel = np.ones((2, 2),np.uint8)
    img_dilation = cv2.dilate(final, kernel, iterations=5)
    erosion = cv2.erode(img_dilation, kernel, iterations = 3)

    final = cv2.circle(erosion, center_coordinates, radius, color, thickness)    

    i = 0

    for x in range(0, width, 12):
        try:
            for y in range(0, height, 12):
                if (final[x, y].any() > 0):

                    i += 1

        except:
            pass

    display = 'Colonies counted: ' + str(i)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(nemo_std)
    axarr[0].set_title('Before segmentation')
    axarr[1].imshow(final)
    axarr[1].set_title('After segmentation')

    plt.text(175, 1000, display, style='italic', bbox={
        'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

    plt.show()
