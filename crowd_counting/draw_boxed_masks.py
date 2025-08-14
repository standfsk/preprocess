import cv2
import glob
import os
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path


def empty_directory(pth):
    if os.path.exists(pth):
        shutil.rmtree(pth)
    os.makedirs(pth, exist_ok=True)


def euclidean_dist(test_matrix, train_matrix):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    num_test = test_matrix.shape[0]
    num_train = train_matrix.shape[0]
    dists = np.zeros((num_test, num_train))
    d1 = -2 * np.dot(test_matrix, train_matrix.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(test_matrix), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(train_matrix), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting

    return dists

if __name__ == "__main__":
    input_path = Path("dataset/fire/val")
    output_path = Path("res")
    empty_directory(output_path)
    image_paths = sorted(input_path.glob("train/*.jpg"))
    pbar = tqdm(image_paths, total=len(image_paths))
    for image_path in tqdm(image_paths):
        pbar.set_description(f"Drawing {image_path}")
        image = cv2.imread(str(image_path))

        label_path = image_path.with_suffix(".npy")
        label_data = np.load(label_path)

        w = image.shape[1]
        h = image.shape[0]
        x_len, y_len = 15, 15

        mask_map = np.zeros((h,w), dtype='uint8')
        centroid_list = []
        wh_list = []
        for point in label_data:
            x, y = min(point[0], w - 1), min(point[1], h - 1)
            x1, y1 = max(int(x - x_len / 2), 0), max(int(y - y_len / 2), 0)
            x2, y2 = min(int(x + x_len / 2), w - 1), min(int(y + y_len / 2), h - 1)

            centroid_list.append([(x2 + x1) / 2, (y2 + y1) / 2])
            wh_list.append([max((x2 - x1) / 2, 3), max((y2 - y1) / 2, 3)])

        centroids = np.array(centroid_list.copy(), dtype=int)
        wh = np.array(wh_list.copy(), dtype=int)
        wh[wh > 25] = 25
        human_num = len(label_data)
        for point in centroids:
            point = point[None, :]
            dists = euclidean_dist(point, centroids)
            dists = dists.squeeze()
            id = np.argsort(dists)

            for start, first in enumerate(id, 0):
                if start > 0 and start < 5:
                    src_point = point.squeeze()
                    dst_point = centroids[first]
                    src_w, src_h = wh[id[0]][0], wh[id[0]][1]
                    dst_w, dst_h = wh[first][0], wh[first][1]

                    count = 0
                    if (src_w + dst_w) - np.abs(src_point[0] - dst_point[0]) > 0 and (src_h + dst_h) - np.abs(
                            src_point[1] - dst_point[1]) > 0:
                        w_reduce = ((src_w + dst_w) - np.abs(src_point[0] - dst_point[0])) / 2
                        h_reduce = ((src_h + dst_h) - np.abs(src_point[1] - dst_point[1])) / 2
                        threshold_w, threshold_h = max(-int(max(src_w - w_reduce, dst_w - w_reduce) / 2.), -60), max(
                            -int(max(src_h - h_reduce, dst_h - h_reduce) / 2.), -60)

                    else:
                        threshold_w, threshold_h = max(-int(max(src_w, dst_w) / 2.), -60), max(
                            -int(max(src_h, dst_h) / 2.), -60)
                    # threshold_w, threshold_h = -5,-5
                    while (src_w + dst_w) - np.abs(src_point[0] - dst_point[0]) > threshold_w and (
                            src_h + dst_h) - np.abs(src_point[1] - dst_point[1]) > threshold_h:

                        if (dst_w * dst_h) > (src_w * src_h):
                            wh[first][0] = max(int(wh[first][0] * 0.9), 1)
                            wh[first][1] = max(int(wh[first][1] * 0.9), 1)
                            dst_w, dst_h = wh[first][0], wh[first][1]
                        else:
                            wh[id[0]][0] = max(int(wh[id[0]][0] * 0.9), 1)
                            wh[id[0]][1] = max(int(wh[id[0]][1] * 0.9), 1)
                            src_w, src_h = wh[id[0]][0], wh[id[0]][1]

                        if human_num >= 3:
                            dst_point_ = centroids[id[start + 1]]
                            dst_w_, dst_h_ = wh[id[start + 1]][0], wh[id[start + 1]][1]
                            if (dst_w_ * dst_h_) > (src_w * src_h) and (dst_w_ * dst_h_) > (dst_w * dst_h):
                                if (src_w + dst_w_) - np.abs(src_point[0] - dst_point_[0]) > threshold_w and (
                                        src_h + dst_h_) - np.abs(src_point[1] - dst_point_[1]) > threshold_h:
                                    wh[id[start + 1]][0] = max(int(wh[id[start + 1]][0] * 0.9), 1)
                                    wh[id[start + 1]][1] = max(int(wh[id[start + 1]][1] * 0.9), 1)

                        count += 1
                        if count > 40:
                            break
            # cv2.circle(image, point, 5, (0, 0, 255), -1)

        for (center_w, center_h), (width, height) in zip(centroids, wh):
            assert (width > 0 and height > 0)

            if (0 < center_w < w) and (0 < center_h < h):
                h_start = (center_h - height)
                h_end = (center_h + height)

                w_start = center_w - width
                w_end = center_w + width
                #
                if h_start < 0:
                    h_start = 0

                if h_end > h:
                    h_end = h

                if w_start < 0:
                    w_start = 0

                if w_end > w:
                    w_end = w
                mask_map[h_start:h_end, w_start: w_end] = 1

        mask_map = mask_map * 255
        # cv2.imwrite(f"res/mask_{os.path.basename(image_path)}", mask_map)
        cv2.putText(image, f"count: {len(label_data)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
        mask_map_color = cv2.cvtColor(mask_map, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(image, 1, mask_map_color, 0.5, 0)
        cv2.imwrite(f"res/{os.path.basename(image_path)}", overlay)
