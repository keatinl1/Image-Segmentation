import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ColonyCounter:

    def __init__(self):
        # 0 - parse folder
        self.path = './'
        self.images = sorted(glob.glob(f'{self.path}/*.jpeg'))

        # upper and lower filter colours
        self.light_orange = (25,20,200)
        self.dark_orange = (220, 240, 235)

        self.radius = 570
        self.color = (0, 0, 0)
        self.thickness = 500        

        self.main()
        
    def preprocessing_function(self, final, center_coordinates):
        '''
            need black spots on white for the counter to work
            hence we invert the colours

        '''

        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(final, kernel, iterations= 5)
        eroded = cv2.erode(dilated, kernel, iterations= 3)
        
        rgb_colonies_alone = cv2.circle(eroded, center_coordinates, self.radius, self.color, self.thickness)
        grey_colonies_alone = cv2.cvtColor(rgb_colonies_alone, cv2.COLOR_RGB2GRAY)

        inverse_grey_colonies = cv2.subtract(255, grey_colonies_alone)

        return inverse_grey_colonies, rgb_colonies_alone

    def counter_function(self, grey_final):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.09
        params.filterByConvexity = True
        params.minConvexity = 0.02
        params.filterByInertia = True
        params.minInertiaRatio = 0.001
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(grey_final)

        return len(keypoints)

    def main(self):

        for image in self.images:

            # 1 - imports and conversions
            original_image = cv2.imread(image)
            hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv_original, self.light_orange, self.dark_orange)
            masked_hsv = cv2.bitwise_and(hsv_original, hsv_original, mask=mask)
            masked_rgb = cv2.cvtColor(masked_hsv, cv2.COLOR_HSV2RGB)

            height, width, _ = masked_rgb.shape
            center_coordinates = (int(width/2), int(height/2))


            # 2 - preprocessing
            inverse_grey_colonies, rgb_colonies_alone = self.preprocessing_function(masked_rgb, center_coordinates)


            # 3 - counting
            number_of_colonies = self.counter_function(inverse_grey_colonies)


            # 4 - plotting
            figure, sub_fig_arr = plt.subplots(1,2)
            figure.set_size_inches(15, 7.5)
            
            sub_fig_arr[0].imshow(original_image)
            sub_fig_arr[0].set_title('Before segmentation')
            sub_fig_arr[1].imshow(rgb_colonies_alone)
            sub_fig_arr[1].set_title('After segmentation')
            
            display = 'Colonies counted: ' + str(number_of_colonies)
            plt.text(300, 1000, display, style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

            plt.show()


if __name__ == "__main__":

    ColonyCounter()
