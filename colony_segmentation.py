import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 0 - parse folder
    path = './'
    images = sorted(glob.glob(f'{path}/*.jpeg'))

    for image in images:

        # 1 - declarations
        nemo_std = cv2.imread(image)
        nemo = cv2.cvtColor(nemo_std, cv2.COLOR_BGR2HSV)

        light_orange = (25,20,200)
        dark_orange = (35, 80, 230)

        mask = cv2.inRange(nemo, light_orange, dark_orange)
        result = cv2.bitwise_and(nemo, nemo, mask=mask)
        final = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

        height, width, depth = final.shape
        center_coordinates = (int(width/2), int(height/2))

        radius = 570
        color = (0, 0, 0)
        thickness = 500


        # 2 - preprocessing
        kernel = np.ones((2, 2),np.uint8)
        img_dilation = cv2.dilate(final, kernel, iterations=5)
        erosion = cv2.erode(img_dilation, kernel, iterations = 3)
        final = cv2.circle(erosion, center_coordinates, radius, color, thickness)
        
        grey_final = cv2.cvtColor(final, cv2.COLOR_RGB2GRAY)
        grey_final = cv2.subtract(255, grey_final)


        # 3 - counting
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.09
        params.filterByConvexity = True
        params.minConvexity = 0.02
        params.filterByInertia = True
        params.minInertiaRatio = 0.001
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(grey_final)
                
        number_of_blobs = len(keypoints)


        # 4 - plotting
        display = 'Colonies counted: ' + str(number_of_blobs)

        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(nemo_std)
        axarr[0].set_title('Before segmentation')
        axarr[1].imshow(final)
        axarr[1].set_title('After segmentation')
        f.set_size_inches(15, 7.5)

        plt.text(300, 1000, display, style='italic', bbox={
            'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.show()


if __name__ == "__main__":
    main()
