# -*- encoding: utf-8 -*-
"""
@File    : tools.py
@Time    : 11/13/2021 9:26 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import cv2
import matplotlib
import numpy as np
import os
from PIL import Image
import numba as nb
import time
import imutils


def normalize_img(uint16_img_original, save_img_name=''):
    uint16_img = uint16_img_original.copy()
    uint16_img = (uint16_img - uint16_img.min()) / (uint16_img.max() - uint16_img.min())
    uint16_img *= 255
    new_uint16_img = uint16_img.astype(np.uint8)
    if 'bilateral' in save_img_name:
        bilateral_dst = cv2.bilateralFilter(new_uint16_img, 9, 75, 75)
        new_uint16_img = bilateral_dst
    # cv2.imshow('UINT16', new_uint16_img)
    # cv2.imwrite('result_' + save_img_name + '.bmp', new_uint16_img)
    return new_uint16_img


def unnormalize_img(uint8_img_original, save_img_name=''):
    uint16_img = uint8_img_original.copy().astype(np.uint16)
    uint16_img = (uint16_img - uint16_img.min()) / (uint16_img.max() - uint16_img.min())
    uint16_img *= 4095
    uint16_img = uint16_img.astype(np.uint16)
    return uint16_img


def read_raw_dr_img(img_file):

    img_name = os.path.basename(img_file).split('.')[0]
    with open(img_file, "rb") as f:
        start_t = time.time()
        width = int.from_bytes(f.read(4), byteorder='little', signed=False)
        height = int.from_bytes(f.read(4), byteorder='little', signed=False)

        img_arr = np.zeros((height, width), dtype=np.uint16)
        for i in range(height):
            for j in range(width):
                x = f.read(2)
                # v = int.from_bytes(x, byteorder='little', signed=False)
                # w = int.from_bytes(x, byteorder='big', signed=True)
                # print(v)
                # print(x)
                # print(x[0])
                # print(x[1])
                # img_arr[i][j] = x[0] + (x[1] % 16) * 256
                img_arr[i][j] = x[0] + (x[1] & 0x0f) * 256
                # print(img_arr[i][j])
        end_t = time.time()

        # print("max value:", np.max(img_arr))
        # print("min value:", np.min(img_arr))
        # np.save('img_arr.npy', img_arr)
        # normalize_img(img_arr, img_name)
        # tiff_img = Image.fromarray(np.uint16(img_arr), mode='L')
        # tiff_img.save("result.tiff")
        # gray_level_reverse(img_arr, img_name)
        print('total read time:', end_t - start_t)
        return img_arr


def read_raw_dr_img_faster(img_file):

    img_name = os.path.basename(img_file).split('.')[0]
    with open(img_file, "rb") as f:
        all_bytes = f.read()
        start_t = time.time()
        width = int.from_bytes(all_bytes[0:4], byteorder='little', signed=False)
        height = int.from_bytes(all_bytes[4:6], byteorder='little', signed=False)

        all_img_bytes = all_bytes[6:]
        print('time for directly reading:', time.time() - start_t)
        # assert len(all_img_bytes) == width * height * 2
        img_arr_val = [ ( all_img_bytes[i:i+1][0] + (all_img_bytes[i+1:i+2][0] & 0x0f) * 256)
                        for i in range(width * height * 2) if i % 2 == 0]
        # print('img arr val', img_arr_val)
        # img_arr_val = [ ( all_img_bytes[i:i+2][0] + all_img_bytes[i:i+2][1]) for i in range(width * height) if i%2 == 0]
        img_arr = np.asarray(img_arr_val, dtype=np.uint16).reshape((height, width))

        end_t = time.time()

        # print("max value:", np.max(img_arr))
        # print("min value:", np.min(img_arr))
        # np.save('img_arr.npy', img_arr)
        # normalize_img(img_arr, img_name)
        # tiff_img = Image.fromarray(np.uint16(img_arr), mode='L')
        # tiff_img.save("result.tiff")
        # gray_level_reverse(img_arr, img_name)
        print('total read time:', end_t - start_t)
        return img_arr


def save_raw_img(img_arr, img_name):
    assert len(img_arr.shape) == 2
    width = img_arr.shape[-1]
    height = img_arr.shape[0]
    img_width_bytes = width.to_bytes(4, byteorder='little', signed=False)
    img_height_bytes = height.to_bytes(4, byteorder='little', signed=False)
    try:
        with open(img_name, 'wb') as f:
            f.write(img_width_bytes)
            f.write(img_height_bytes)
            for i in range(height):
                for j in range(width):
                    pixel_val_bytes = int(img_arr[i][j]).to_bytes(2, byteorder='little', signed=False)
                    f.write(pixel_val_bytes)
        print('Image save successfully! ')
    except Exception as e:
        print(e)
        return False
    return True


def gray_level_transformation(img_arr_original, window_level, window_width, method='linear'):
    img_arr = img_arr_original
    # min_gray_level = 0
    # max_gray_level = 4095
    min_gray_level = np.min(img_arr)
    max_gray_level = np.max(img_arr)
    print('*' * 100)
    print('all zeros ???', np.all(img_arr == 0))
    print('in func:', window_level, ":", window_width)
    print('in gray level transformation', min_gray_level, ":", max_gray_level)

    # if not ( window_level >= min_gray_level and window_level <= max_gray_level):
    #     print(not ( window_level >= min_gray_level and window_level <= max_gray_level))
    #     print('window level', window_level)
    #     print(window_level, 'is a invalid window level value')
    #     return
    # if window_width > min(window_level - min_gray_level, max_gray_level - window_level) * 2:
    #     print(window_width > min(window_level - min_gray_level, max_gray_level - window_level) * 2)
    #     print('window width:', window_width)
    #     print(window_width, 'is a invalid window width value')
    #     return

    gray_level_start = window_level - window_width / 2
    gray_level_end = window_level + window_width / 2
    if method == 'linear':
        # for i in range(img_arr.shape[0]):
        #     for j in range(img_arr.shape[-1]):
        #         if img_arr[i][j] > gray_level_end:
        #             img_arr[i][j] = 255
        #         elif img_arr[i][j] < gray_level_start:
        #             img_arr[i][j] = 0
        #         else:
        #             img_arr[i][j] = ((img_arr[i][j] - gray_level_start) / ( gray_level_end - gray_level_start)) * 4095

        print('all small ?', np.all(img_arr < gray_level_start))
        print('all big ?', np.all(img_arr > gray_level_end))
        print('all in ?', np.all( (img_arr >= gray_level_start) & (img_arr <= gray_level_end)) )
        conditions = [img_arr > gray_level_end, img_arr < gray_level_start,
                      (img_arr >= gray_level_start) & (img_arr <= gray_level_end) ]
        choices = [255, 0, (lambda x:( (x - gray_level_start) / float((gray_level_end - gray_level_start)) ) * 255)(img_arr)]
        img_arr = np.select(conditions, choices)
        print('finished')
        print('all zeros ?', np.all(img_arr == 0))
        print('min img arr:', np.min(img_arr))
        print('max img arr:', np.max(img_arr))
    # print('img_arr is', img_arr)
    print('*' * 100)
    return img_arr


def dr_img_details_enhancing(img_arr):
    pass


def gray_level_reverse(img_arr, img_name):
    img_arr = 4095 - img_arr
    normalize_img(img_arr, img_name + '_gray_reversion')
    pass

def square_transform(img_arr):
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[-1]):
            img_arr[i][j] = pow(img_arr[i][j], 2) / 255
    return img_arr

# 这个是可以选用的点
def log_transform(img_arr):
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[-1]):
            img_arr[i][j] = np.log(img_arr[i][j] + 1) * 46
    return img_arr


# 这里的像素值需要进行归一化吗？为什么两次看到的公式都不一样？
# 这里只需要用在那个区间值的映射里面就行
def log_transform2(img_arr, coefficient_c, coefficient_v):
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[-1]):
            img_arr[i][j] = (np.log(img_arr[i][j] * coefficient_v + 1) / np.log(coefficient_v + 1)) * coefficient_c
    return img_arr


def gamma_transform(img_arr):
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[-1]):
            img_arr[i][j] = 6.031e-8 * pow(img_arr[i][j], 4)
    return img_arr


# def gray_histogram_2(img_arr, is_densed=False, is_cumulative=False):
#     number_of_bins = 4096
#
#     fig = pyplot.figure()
#     plt = fig.add_subplot(111)
#     # 根据像素灰度统计结果来显示灰度直方图
#     histogram, bins, patch = plt.hist(img_arr.flatten(), number_of_bins,
#                                       facecolor='blue', histtype='bar',
#                                       cumulative=is_cumulative, density=is_densed
#                                       )
#     print('hists:', histogram)
#     print('max:', np.max(histogram))
#     plt.xlabel('gray level')
#     if is_densed:
#         plt.ylabel('pixel ratio')
#     else:
#         plt.ylabel('number of pixels')
#     plt.axis([0, 4500, 0, np.max(histogram)])
#     title = ('Cumulative 'if is_cumulative else '') + 'Histogram Of Image'
#     plt.title(title)
#     # plt.savefig('img_gray_histogram.png')
#     # plt.show()
#     return fig


def gray_histogram(img_arr):
    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    plt = fig.add_subplot(111)

    # 根据像素灰度统计结果来显示灰度直方图
    hist, bins = np.histogram(img_arr.flatten(), 4096, [0, 4096])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # plt.plot(cdf_normalized, color='b')
    plt.hist(img_arr.flatten(), 4096, [0, 4096], color='r', density=False, cumulative=False, rwidth=35)
    # plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.axis([0, 4096, 0, 10000])
    plt.legend(('histogram'), loc='upper left')
    # plt.savefig('histogram2.png')
    # plt.show()
    return fig


def get_plt():
    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    # fig = matplotlib.pyplot.figure()
    t = np.arange(0, 3, .01)
    pt = fig.add_subplot(111)
    pt.plot(t, 2 * np.sin(2 * np.pi * t))

    # plt.plot(t, 2 * np.sin(2 * np.pi * t))
    return fig


def save_raw_img(img_arr, img_name):
    assert len(img_arr.shape) == 2
    width = img_arr.shape[-1]
    height = img_arr.shape[0]
    img_width_bytes = width.to_bytes(4, byteorder='little', signed=False)
    img_height_bytes = height.to_bytes(4, byteorder='little', signed=False)
    try:
        with open(img_name, 'wb') as f:
            f.write(img_width_bytes)
            f.write(img_height_bytes)
            for i in range(height):
                for j in range(width):
                    pixel_val_bytes = int(img_arr[i][j]).to_bytes(2, byteorder='little', signed=False)
                    f.write(pixel_val_bytes)
        print('Image save successfully! ')
    except Exception as e:
        print(e)
        return False
    return True


def gray_histogram_equalization(img_arr):
    # 根据像素灰度统计结果来显示灰度直方图
    hist, bins = np.histogram(img_arr.flatten(), 4096, [0, 4095])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    print('cdf img arr', img_arr)
    print('cdf img arr dtype:', img_arr.dtype)
    equalized_img = cdf[img_arr]
    equalized_img = equalized_img.astype('uint16')
    print('max equalized img:', equalized_img.max())
    print('min equalized img:', equalized_img.min())
    return equalized_img
    # equalized_img_2 = cv2.equalizeHist(img_arr) 此函数无法在此处使用


def clahe(img_arr):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img_arr)
    return cl1


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # nW = w
    # nH = h
    return cv2.warpAffine(image, M, (nW, nH))


def homomorphic_filter(original_img_arr, settings=dict(d0=10, r1=0.5, rh=2, c=4, h=2, l=0.6)):

    rows, cols = original_img_arr.shape
    original_img_arr_log = np.log(original_img_arr + 1)
    original_img_arr_fft = np.fft.fft2(original_img_arr_log)
    original_img_arr_fftshift = np.fft.fftshift(original_img_arr_fft)
    print('min', np.min(original_img_arr_fftshift))
    print('max', np.max(original_img_arr_fftshift))
    original_img_arr_dst_fftshift = np.zeros_like(original_img_arr_fftshift)

    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows//2, rows//2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (settings['rh'] - settings['r1']) *\
        (1 - np.exp(-settings['c'] * (D ** 2 / settings['d0'] ** 2))) + settings['r1']
    print('Z is', Z)
    original_img_arr_dst_fftshift = Z * original_img_arr_fftshift
    original_img_arr_dst_fftshift = (settings['h'] - settings['l']) * original_img_arr_dst_fftshift + settings['l']
    original_img_arr_dst_fftshift = np.fft.ifftshift(original_img_arr_dst_fftshift)
    original_img_arr_dst_ifft = np.fft.ifft2(original_img_arr_dst_fftshift)
    dst = np.exp(np.real(original_img_arr_dst_ifft))
    dst = np.uint8(np.clip(dst, 0, 255))
    print('dst min:', np.min(dst))
    print('dst max:', np.max(dst))
    return dst


# if __name__ == '__main__':
#     file_list = ['raws/lumbar.raw',
#             'raws/lung.raw',
#             'raws/vertebra.raw']
#     file_idx = 2
#     file = file_list[file_idx]
#     img_name = os.path.basename(file).split('.')[0]
#     img_arr = read_raw_dr_img(file)

    # img_arr = gray_level_transformation(img_arr, window_level=2000, window_width=2000)
    # normalize_img(img_arr, img_name + '_transformed')

    # img_arr = square_transform(img_arr)
    # normalize_img(img_arr, img_name + '_squared')

    # img_arr = log_transform(img_arr)
    # normalize_img(img_arr, img_name + '_log')

    # img_arr = gamma_transform(img_arr)
    # normalize_img(img_arr, img_name + '_gamma')
    # gray_histogram(img_arr, is_densed=False, is_cumulative=True)

    # gs_dst = cv2.GaussianBlur(img_arr, (5, 5), cv2.BORDER_DEFAULT)
    # normalize_img(gs_dst, img_name + '_gs')
    # cv2.imshow('Gaussian SMoothing ', np.hstack((img_arr, dst)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # avg_dst = cv2.blur(img_arr, (5, 5))
    # normalize_img(avg_dst, img_name + '_avg')

    # median_dst = cv2.medianBlur(img_arr, 5)
    # normalize_img(median_dst, img_name + '_median')

    # normalize_img(img_arr, img_name + '_bilateral')

    # gray_histogram_2(img_arr)
    # equalized_img = gray_histogram_equalization(img_arr)
    # normalize_img(equalized_img, img_name + '_equalized')
    #
    # equalized_img = clahe(img_arr)
    # normalize_img(equalized_img, img_name + '_clahe')
