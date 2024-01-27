import cv as cv
import matplotlib.pyplot as plt
import numpy as np
import random 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Import image
def read_image(path):
    img = cv.imread(path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray_img

def harris_corner_detector(img, gray_img):
    gray_img = np.float32(gray_img)
    height = img.shape[0]
    width = img.shape[1]
    matrix_r = np.zeros((height, width))
    
    img_gaussian = cv.GaussianBlur(gray_img, (3, 3), 0)

    ix = cv.Sobel(img_gaussian, cv.CV_64F, 1, 0, ksize=3)
    iy = cv.Sobel(img_gaussian, cv.CV_64F, 0, 1, ksize=3)

    ixx = np.square(ix)
    iyy = np.square(iy)
    ixy = ix * iy

    window_size = 2
    offset = int(window_size / 2)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            sxx = np.sum(ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            syy = np.sum(iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            sxy = np.sum(ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            h = np.array([[sxx, sxy], [sxy, syy]])

            det = np.linalg.det(h)
            tr = np.matrix.trace(h)
            k = 0.04

            r = det - k * (tr**2)
            matrix_r[y - offset, x - offset] = r

    dst = matrix_r

    result_img = img.copy()
    result_img[dst > 0.01 * dst.max()] = [0, 0, 255]
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv.KeyPoint(float(x[1]), float(x[0]), 13) for x in keypoints]

    return (keypoints, result_img)

def harris_feature_extraction(img, gray_img):
    gray_img = np.float32(gray_img)
    dst = cv.cornerHarris(gray_img, 2, 3, 0.04)
    result_img = img.copy()
    result_img[dst > 0.01 * dst.max()] = [0, 0, 255]
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv.KeyPoint(float(x[1]), float(x[0]), 13) for x in keypoints]
    return (keypoints, result_img)

def sift_compute(img, kp):
    sift = cv.SIFT_create()
    kp, descriptors = sift.compute(img, kp)
    return kp, descriptors

def sift_detect_and_compute(img):
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def feature_matcher(kp1, des1, kp2, des2, threshold):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches


def calculate_homography_matrix(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H / H[2, 2]  # standardize to let w*H[2,2] = 1
    return H

def select_random_points(matches):
    idx = random.sample(range(len(matches)), 4)
    point = [matches[i] for i in idx]
    return np.array(point)

def calculate_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp / temp[2])[0:2]
    errors = np.linalg.norm(all_p2 - estimate_p2, axis=1) ** 2
    return errors

def ransac_algorithm(matches, threshold, iterations):
    num_best_inliers = 0

    for i in range(iterations):
        points = select_random_points(matches)
        H = calculate_homography_matrix(points)

        if np.linalg.matrix_rank(H) < 3:
            continue

        errors = calculate_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy

    print("Inliers/Matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H




#Harris --> failure: 02-->error, 03-->이상함 걍 
img, gray_img = read_image("1.png")
img_2, gray_img_2 = read_image("3.png")


# keypoints, result_img = harris(img, gray_img)
# keypoints_2, result_img_2 = harris(img_2, gray_img_2)
# keypoints, descriptors = siftCompute(gray_img,keypoints)
# keypoints_2, descriptors_2 = siftCompute(gray_img_2,keypoints_2)

keypoints, descriptors = sift_detect_and_compute(gray_img)
keypoints_2, descriptors_2 = sift_detect_and_compute(gray_img_2)






# cv.imshow('result_img',result_img)
# cv.imshow('result_img_2',result_img_2)
# cv.waitKey(0)


matches = feature_matcher(keypoints, descriptors, keypoints_2, descriptors_2, 0.70)
print(len(matches))

# # total_img = np.concatenate((img, img_2), axis=1)
# inliers, H = ransac_algorithm(matches, 0.5, 2000)
# # plot_matches(inliers, total_img) 
# cv.imshow('stitched image',stitch_img(img, img_2, H))





