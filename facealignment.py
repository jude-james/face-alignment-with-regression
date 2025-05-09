import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Load the train data using np.load
train_data = np.load('face_alignment_training_images.npz', allow_pickle=True)

# Extract the training images and points
train_images = train_data['images']
train_points = train_data['points']

# Load the test data
test_data = np.load('face_alignment_test_images.npz', allow_pickle=True)

# Extract the testing images
test_images = test_data['images']

# Preprocess an images array
def preprocess(images, scale):
    preprocess_images = []
    for img in images:
        # Resize the image to the new scale
        resized_img = cv2.resize(img, [int(img.shape[0] * scale), int(img.shape[1] * scale)])
        # Convert to greyscale for SIFT
        greyscaled_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        preprocess_images.append(greyscaled_img)
    return np.array(preprocess_images)

# Resize an array of points by the scale
def resize_points(pointsArr, scale):
    resized_pointsArr = []
    for points in pointsArr:
        resized_pointsArr.append(points[:,:] * scale)
    return np.array(resized_pointsArr)

resize_scale = 0.25 # Used for all images and points

# Preprocess the train and test images
train_images_preprocessed = preprocess(train_images, resize_scale)
test_images_preprocessed = preprocess(test_images, resize_scale)

# Then resize the train points to match the new image size
train_points_resized = resize_points(train_points, resize_scale)

# define my own keypoints for SIFT to create descriptors (using original image scale of 256x256)
predefined_points = np.array([
    [85, 100],  # left eye
    [170, 100],  # right eye
    [128, 140],  # nose tip
    [95, 180],  # left mouth corner
    [160, 180]   # right mouth corner
], dtype=np.float32)

# Then again resize these points to match the scale
for point in predefined_points:
    point[0] = point[0] * resize_scale
    point[1] = point[1] * resize_scale

print("predefined points:\n", predefined_points)

# create sift object
sift = cv2.SIFT_create()

def compute_descriptors(image, points):
    size = 100 * resize_scale
    keypoints = [cv2.KeyPoint(float(x), float(y), size) for (x, y) in points]
    # Use sift.compute at the keypoint
    keypoints, descriptors = sift.compute(image, keypoints)  
    return descriptors

x_train = [] # sift descriptors
y_train = [] # training data points 

for img, points in zip(train_images_preprocessed, train_points_resized):
    descriptors = compute_descriptors(img, points)
    x_train.append(descriptors.flatten())
    y_train.append(points.flatten())

x_train = np.array(x_train)
y_train = np.array(y_train)

# Create and fit the model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# TODO cascaded regression function here...
# Loop through test images to get descriptors for model predictions
x_test = []
for img in test_images_preprocessed:
    descriptors = compute_descriptors(img, predefined_points)
    x_test.append(descriptors.flatten())

x_test = np.array(x_test)
# Run prediction on array of descriptors for all test images 
prediction_points = model.predict(x_test).reshape(-1, 5, 2)
print("final predictions shape:", prediction_points.shape)

def visualise_pts_2(img, pts, pts2):
    plt.imshow(img, cmap='gray')
    plt.plot(pts[:, 0], pts[:, 1], '+r')
    plt.plot(pts2[:, 0], pts2[:, 1], '+g')
    plt.show()

'''
for i in range(200, 201):
    visualise_pts_2(test_images_preprocessed[i], prediction_points[i], predefined_points)
'''

def visualise_pts(img, pts):
    plt.imshow(img, cmap='gray')
    plt.plot(pts[:, 0], pts[:, 1], '+r')
    plt.show()

'''
for i in range(10):
    idx = np.random.randint(0, train_images_preprocessed.shape[0])
    visualise_pts_2(train_images_preprocessed[idx, ...], train_points_resized[idx, ...], predefined_points)
'''
    
# This function will draw the keypoints on the images, showing their orientation
def draw_im_kps(img, kps):  
    image_with_kp = cv2.drawKeypoints(img, kps, None, flags=4)
    plt.imshow(image_with_kp, cmap='gray')
    plt.title("SIFT keypoints")
    plt.show()

# TODO use this for evaluating data
def euclid_dist(pred_pts, gt_pts):
    """
    Calculate the euclidean distance between pairs of points
    :param pred_pts: The predicted points
    :param gt_pts: The ground truth points
    :return: An array of shape (no_points,) containing the distance of each predicted point from the ground truth
    """
    import numpy as np
    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))

# TODO finish save as csv function, check output file is correct
def save_as_csv(points, location = '.'):
        """
        Save the points out as a .csv file
        :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
        :param location: Directory to save results.csv in. Default to current working directory
        """
        assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
        assert np.prod(points.shape[1:])==5*2, 'wrong number of points provided. There should be 5 points with 2 values (x,y) per point'
        np.savetxt(location + '/results_task2.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

# Resize the prediction points back to the original size
prediction_points_resized = resize_points(prediction_points, 1 / resize_scale)
save_as_csv(prediction_points_resized)