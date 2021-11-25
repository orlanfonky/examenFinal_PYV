import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

font = cv2.FONT_HERSHEY_SIMPLEX

# Method used to show an image in a resizable window
def show_image(window_name, image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)


# Method used to update the image in a windows that has associated an event handler
def update_image_window(points_list, counter, image_draw):
    if len(points_list) > counter:
        counter = len(points_list)
        cv2.circle(image_draw, (points_list[-1][0], points_list[-1][1]), 3, [0, 0, 255], -1)
        labelClickedPixel(image_draw, points_list[-1][0], points_list[-1][1])
    return counter


# Method used to handle the click event in an image
def labelClickedPixel(target_image, x, y):
    cv2.putText(target_image, str(x) + ',' +
                str(y), (x, y), font,
                1, (255, 0, 0), 2)

class ImageManager:

    # Init method, used to clear all the class attributes
    def __init__(self):
        self.image = None
        self.seed_points = []
        self.bounding_box_points = []

    # Method used to load the image found in the image_path parameter, the method takes into account if the image
    # that is going to be loaded is a DICOM image or a normal image
    def set_image(self, image_path):
        self.image = cv2.imread(image_path)


    def process_image(self):
        if self.image is not None:
            self.color_porcent()
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            image_draw = np.copy(self.image)
            cv2.setMouseCallback("Image", self.process_click_event, image_draw)

            self.seed_points = []
            print("Fisrt choose two points to drawn the line")
            print("Then choose a third point to dranw a parale line to the line formed betwen the firstsones points choosed")
            bounding_box_counter = 0
            while len(self.bounding_box_points) < 3:
                cv2.imshow("Image", image_draw)
                cv2.waitKey(1)
                bounding_box_counter = update_image_window(self.bounding_box_points, bounding_box_counter, image_draw)

            labelClickedPixel(image_draw, self.seed_points[-1][0], self.seed_points[-1][1])

            # Crop the image to the rectangle defined
            x1, y1 = self.bounding_box_points[0]
            x2, y2 = self.bounding_box_points[1]
            x3, y3 = self.bounding_box_points[2]

            cv2.line(image_draw, (x1,y1),(x2,y2), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(image_draw, (x1,y1),(x2,y2), (255, 0, 0), 1, cv2.LINE_AA)

            pend1 = ((y2 - y1) / (x2 - x1) + 1e-8)
            if pend1 < 0:
                pend1 = pend1 * (-1)
            # tenemos la pendiente, nesecitamos el corte con el EJE "B"
            corteEje = int(y1 - (pend1 * x1))
# A PARTIR DE LA PENDIENTE CALCULAMOS las coordenadas para el segundo punto de la segunda recta
            x4 = x3 + 100
          #  print("DATA: ", x4," pend:",pend1, "corte",corteEje)
            y4 = int((pend1 * (x4 + 100)) + corteEje)

            cv2.line(image_draw, (x3,y3), ((x1 + 100), y4), (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow("RESULT 4.", image_draw)
            cv2.waitKey(0)
            print("PUNTO 1:")
            self.color_porcent()


    # Method used to delete the image loaded in memory
    def unload_image(self):
        if self.image is not None:
            self.image = None
            self.image_type = None

        else:
            print("Current image is empty")

    # Method to handle a click event in the image window
    def process_click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.seed_points.append((x, y))
            self.bounding_box_points.append((x, y))

    def porcentaje(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        n_colors = 4
        method = ['kmeans', 'gmm']
        select = 1
        image = np.array(image, dtype=np.float64) / 255
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))

        print("Fitting model on a small sub-sample of the data")
      # model = GMM(n_components=n_colors).fit(image_array_sample)
        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

        if method[select] == 'gmm':
            labels = model.predict(image_array)
            centers = model.means_
        else:
            labels = model.predict(image_array)
            centers = model.cluster_centers_

        plt.figure(1)
        plt.clf()
        plt.axis('off')
        plt.title('Original image')
        plt.imshow(image)

        plt.figure(2)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} colors, method={})'.format(n_colors, method[select]))
        plt.imshow(self.recreate_image(centers, labels, rows, cols))

        plt.show()

    def recreate_image(self, centers, labels, rows, cols):
        d = centers.shape[1]
        image_clusters = np.zeros((rows, cols, d))
        label_idx = 0
        for i in range(rows):
            for j in range(cols):
                image_clusters[i][j] = centers[labels[label_idx]]
                label_idx += 1
        return image_clusters

    def color_porcent(self):

        imagenHSV = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        verdeBajo = np.array([36, 50, 20], np.uint8)
        verdeAlto = np.array([70, 255, 255], np.uint8)
        #maskVerde = cv2.inRange(imagenHSV, verdeBajo, verdeAlto)
        #contornosVerde = cv2.findContours(maskVerde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # hacemos la mask y filtramos en la original
        mask = cv2.inRange(imagenHSV, verdeBajo, verdeAlto)
        res = cv2.bitwise_and(self.image, self.image, mask=mask)
        print("SHAPE Originial: ", self.image.shape)
       # for i in range(len(mask)):
        #    print("ASDSAD",i)

        plt.subplot(1, 2, 1)
        plt.imshow(mask,cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(res)
        plt.show()
        print("RESSHAPE: ", res.shape)