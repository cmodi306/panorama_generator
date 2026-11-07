import numpy as np
import cv2
import os

class ImageMatcher:
    def __init__(self, detector, img1, img2):
        self.img1 = img1
        self.img2 = img2
        if detector == "sift":
            self.detector = cv2.SIFT_create()
            self.norm = cv2.NORM_L2
        elif detector == "orb":
            self.detector = cv2.ORB_create()
            self.norm = cv2.NORM_HAMMING
        else:
            raise NameError("Please enter a valid detector name.")

    def get_keypoints_and_descriptors(self):
        kp1, des1 = self.detector.detectAndCompute(self.img1, None)
        kp2, des2 = self.detector.detectAndCompute(self.img2, None)
        return kp1, des1, kp2, des2

    def match_descriptors(self, des1, des2, filter=True):    
        matcher = cv2.BFMatcher(normType=self.norm)
        matches = matcher.knnMatch(des1, des2, k=2)
        if filter:
            good_matches = []
            for match in matches:
                if len(match) == 2 and match[0].distance < 0.5 * match[1].distance:
                    good_matches.append(match[0])
            return good_matches
        return matches

    def get_matches(self):
        kp1, des1, kp2, des2 = self.get_keypoints_and_descriptors()
        print(f"Number of keypoints in image 1: {len(kp1)}")
        print(f"Number of keypoints in image 2: {len(kp2)}")
        matches = self.match_descriptors(des1=des1, des2=des2)
        print(f"Number of matches: {len(matches)}")        
        return matches


def main():
    image_1 = "f1_aus_gp_1.jpg" #input("Enter first image path: ")
    image_2 = "f1_aus_gp_2.jpg" #input("Enter second image path: ")

    if not os.path.exists(image_1) or not os.path.exists(image_2):
        raise FileNotFoundError("One or both of the image files not found.")

    image_1 = cv2.imread(image_1)
    image_2 = cv2.imread(image_2)

    image_matcher = ImageMatcher("sift", image_1, image_2)
    matches = image_matcher.get_matches()
    kp1, des1, kp2, des2 = image_matcher.get_keypoints_and_descriptors()
    matched_image = cv2.drawMatches(image_1,kp1,image_2,kp2,matches, \
                                       None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite("panorama_matched.jpg", matched_image)

    # saving the matches pixels from the individual images
    left_kps, right_kps = [], []
    for match in matches:
        left_kps.append(kp1[match.queryIdx].pt)
        right_kps.append(kp2[match.trainIdx].pt)

    left_kps = np.float32((left_kps))
    right_kps = np.float32((right_kps))

    H, mask = cv2.findHomography(right_kps, left_kps, cv2.RANSAC, 5.0)
    print("inliers:", int(mask.sum()), "of", len(matches))

    h, w = image_1.shape[:2]
    warped = cv2.warpPerspective(image_2, H, (w + image_2.shape[1], h))
    panorama = warped.copy()
    panorama[0:h, 0:w] = image_1
    cv2.imwrite("panorama_without_blend.jpg", panorama)
    
    x0 = np.where(warped.sum(axis=2) > 0)[1].min()        # where the right photo starts
    a = np.linspace(0, 1, w - x0).reshape(1, -1, 1)          # 0 = all left, 1 = all right
    L = image_1[:, x0:w].astype(float)
    R = warped[:, x0:w].astype(float)
    blend = L * (1 - a) + R * a
    blend[R.sum(axis=2) == 0] = L[R.sum(axis=2) == 0]       # no right pixel here? keep left
    panorama[:, x0:w] = blend.astype(np.uint8)

    x_end = np.where(panorama.sum(axis=0) > 0)[0].max()
    panorama = panorama[:, :x_end + 1]
    cv2.imwrite("panorama.jpg", panorama)
    
main()