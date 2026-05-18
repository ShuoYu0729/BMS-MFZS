import os
import glob
import cv2
import numpy as np


def contrast_algorithm(input_pattern, output_dir=None):
    """
    Local spatio-temporal contrast candidate selection.

    Args:
        input_pattern: image path pattern, e.g. r"D:/data/*.jpg"
        output_dir: optional directory for saving contrast results

    Returns:
        con_result_array: list of contrast-enhanced grayscale images
        len(con_result_array): number of generated results
    """

    img_array = []
    con_result_array = []

    file_list = sorted(glob.glob(input_pattern))

    for filename in file_list:
        print(f"Reading file: {filename}")
        img_re = cv2.imread(filename)

        if img_re is None:
            print(f"Failed to read file: {filename}")
            continue

        img_gray = cv2.cvtColor(img_re, cv2.COLOR_BGR2GRAY)
        img_array.append(img_gray)

    if not img_array:
        print("No images were read. Please check the file path and file extensions.")
        return con_result_array, 0

    leng = len(img_array)
    print(f"Number of images read: {leng}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    pad = 9
    diliration = 9
    di = int((diliration - 1) / 2)

    def patch_mean(x_start, y_start, img):
        """
        Compute mean gray value of a 3x3 local patch.
        Note:
            img is indexed as img[y, x].
        """
        h, w = img.shape[:2]

        x_start = max(0, min(x_start, w - 3))
        y_start = max(0, min(y_start, h - 3))

        local_patch = img[y_start:y_start + 3, x_start:x_start + 3]
        return float(np.mean(local_patch))

    M = 0

    for i_num in range(len(img_array) - 2):
        img_conti_1 = img_array[i_num].copy()
        img_conti_2 = img_array[i_num + 1].copy()
        img_conti_3 = img_array[i_num + 2].copy()

        h_con, w_con = img_conti_2.shape[:2]

        m = w_con - pad + 1
        n = h_con - pad + 1

        candidate_points = set()

        for y in range(0, n):
            for x in range(0, m):
                center_x = int(x + di)
                center_y = int(y + di)

                for py in range(y, y + pad - 2):
                    for px in range(x, x + pad - 2):

                        if px == center_x and py == center_y:
                            continue

                        sym_x = 2 * center_x - px - 2
                        sym_y = 2 * center_y - py - 2

                        if sym_x < 0 or sym_x + 3 > w_con or sym_y < 0 or sym_y + 3 > h_con:
                            continue

                        center_mid = int(img_conti_2[center_y, center_x])
                        center_fir = int(img_conti_1[center_y, center_x])
                        center_thir = int(img_conti_3[center_y, center_x])

                        mid_di = center_mid - patch_mean(px, py, img_conti_2)
                        mid_condi = center_mid - patch_mean(sym_x, sym_y, img_conti_2)

                        fir_di = center_fir - patch_mean(px, py, img_conti_1)
                        fir_condi = center_fir - patch_mean(sym_x, sym_y, img_conti_1)

                        thir_di = center_thir - patch_mean(px, py, img_conti_3)
                        thir_condi = center_thir - patch_mean(sym_x, sym_y, img_conti_3)

                        comp_t = abs(
                            (fir_di - mid_di) * (thir_di - mid_di)
                            + (fir_condi - mid_condi) * (thir_condi - mid_condi)
                        )

                        comp_s = abs(mid_condi * mid_di)

                        comp_c = 0.3 * comp_s + 0.7 * comp_t

                        # 修正原始 bug:
                        # 原来是 comp_c > 0.3*comp_s + 0.75*comp_t，
                        # 由于 comp_c = 0.3*comp_s + 0.7*comp_t，
                        # 在 comp_t > 0 时基本不可能成立。
                        threshold = 0.3 * comp_s + 0.55 * comp_t

                        if comp_t != 0:
                            if comp_c > threshold and img_conti_2[py, px] != 0:
                                candidate_points.add((px, py))

        result_img = img_conti_2.copy()

        for y_de in range(0, h_con):
            for x_de in range(0, w_con):
                if (x_de, y_de) in candidate_points:
                    result_img[y_de, x_de] = 255
                else:
                    result_img[y_de, x_de] = int(result_img[y_de, x_de]) * 0.1

        M += 1
        print(f"Frame {M}, candidate points: {len(candidate_points)}")

        con_result_array.append(result_img.copy())

        if output_dir is not None:
            save_path = os.path.join(output_dir, f"{M}.jpg")
            cv2.imwrite(save_path, result_img)

    return con_result_array, len(con_result_array)
