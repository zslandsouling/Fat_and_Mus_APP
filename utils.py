import os
import SimpleITK as itk
import numpy as np
import cv2
# import xlwt, xlrd
# import xlutils.copy

def read_nii(nii_path):
    itk_img = itk.ReadImage(nii_path)
    img = itk.GetArrayFromImage(itk_img)
    return img


def ori2new_nii(ori_nii_path, new_mask, new_nii_path):
    itk_img = itk.ReadImage(ori_nii_path)
    out = itk.GetImageFromArray(new_mask)
    out.SetOrigin(itk_img.GetOrigin())
    out.SetSpacing(itk_img.GetSpacing())
    out.SetDirection(itk_img.GetDirection())
    itk.WriteImage(out, new_nii_path)


def detect_y(img):
    y_min = 0
    y_max = 0
    for n in range(img.shape[1]):
        if np.any(img[n, :]):
            y_max = n
            if y_min == 0:
                y_min = n
    return y_min, y_max


def get_max_area_contours2(contours):
    sum = list()
    max_are=0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        sum.append(area)
        sum.sort()
        max_are = sum[-1]
    return max_are


def remove_area(image, kernel, iters=2, alpha=0.5, beta=0.5):

    image = np.uint8(image * 255)
    ct_img, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_ct_img = []
    new_img = image
    if len(ct_img) > 1:
        for c in ct_img:
            (x, y, w, h) = cv2.boundingRect(c)
            if y < 0.75 * image.shape[0]:
                # if w>0:
                new_ct_img.append(c)
    else:
        new_ct_img = ct_img
    cv2.fillPoly(new_img, new_ct_img, 255)
    y_min, y_max = detect_y(image)

    no_img = np.zeros((image.shape))
    erosion = cv2.erode(new_img, kernel, iterations=iters)

    ct_erosion, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    cond_y = y_min + alpha * (y_max - y_min)
    max_area_ct = get_max_area_contours2(ct_erosion)
    cond_area = beta * max_area_ct
    #     if len(new_ct_img) < len(ct_erosion):
    for c in ct_erosion:
        (x, y, w, h) = cv2.boundingRect(c)
        ct_erosion_area = cv2.contourArea(c)
        if y < cond_y and ct_erosion_area > cond_area:
            cv_contours.append(c)
        else:
            continue
    #         cv_contours = ct_erosion
    cv2.fillPoly(no_img, cv_contours, 255)
    rm_img = no_img
    return rm_img


def dilate_2_size(image, kernel, iters):
    image = np.uint8(image * 255)
    img_dilate = cv2.dilate(image, kernel, iterations=iters)
    contours_dil, _ = cv2.findContours(img_dilate,
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(img_dilate, contours_dil, 255)

    return img_dilate


def remove_archs_spine_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    iters = 4
    mask_255 = np.zeros((mask.shape))
    try:
        for k in range(mask.shape[0]):
            img = mask[k, :, :]
            norm_img = img
            rm_img = remove_area(norm_img, kernel=kernel, iters=iters, alpha=0.8, beta=0.5)
            dil_img = dilate_2_size(rm_img, kernel=kernel, iters=iters)
            mask_255[k, :, :] = dil_img
    except:
        for k in range(mask.shape[0]):
            img = mask[k, :, :]
            norm_img = cv2.resize(img, (mask.shape[2], mask.shape[2]), interpolation=cv2.INTER_NEAREST)
            rm_img = remove_area(norm_img, kernel=kernel, iters=iters, alpha=0.8, beta=0.5)
            dil_img = dilate_2_size(rm_img, kernel=kernel, iters=iters)
            dil_img2 = cv2.resize(dil_img, (mask.shape[2], mask.shape[1]), interpolation=cv2.INTER_NEAREST)
            mask_255[k, :, :] = dil_img2

    assert mask_255.shape == mask.shape
    d, h, w = mask_255.shape
    cor_label = np.zeros((d, h, w))

    for i in range(d):
        for j in range(h):
            for l in range(w):
                if mask_255[i, j, l] == 255:
                    cor_label[i, j, l] = mask[i, j, l]
    return cor_label

