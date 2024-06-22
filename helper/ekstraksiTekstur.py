import cv2
import numpy as np
import os
from skimage.feature import graycomatrix

def calculate_glcm_features(glcm):
    # Mendapatkan jumlah level abu-abu
    L = glcm.shape[0]
    
    # Inisialisasi variabel fitur
    asm = 0
    contrast = 0
    idm = 0
    entropy = 0
    correlation = 0

    # Mean dan standar deviasi untuk perhitungan korelasi
    px = np.sum(glcm, axis=1)
    py = np.sum(glcm, axis=0)
    mean_i = np.sum(np.arange(L) * px)
    mean_j = np.sum(np.arange(L) * py)
    std_i = np.sqrt(np.sum((np.arange(L) - mean_i)**2 * px))
    std_j = np.sqrt(np.sum((np.arange(L) - mean_j)**2 * py))

    # Menghitung fitur GLCM
    for i in range(L):
        for j in range(L):
            p_ij = glcm[i, j]
            if p_ij > 0:
                asm += p_ij**2
                contrast += (i - j)**2 * p_ij
                idm += p_ij / (1 + (i - j)**2)
                entropy -= p_ij * np.log2(p_ij)
                correlation += ((i - mean_i) * (j - mean_j) * p_ij) / (std_i * std_j)

    return asm, contrast, idm, entropy, correlation

def ekstrakTekstur(image):
    # Memuat gambar dalam grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize gambar ke ukuran yang diinginkan (misalnya 512x512)
    image_resized = cv2.resize(imageGray, (512, 512))
    
    # Mendefinisikan jarak dan sudut untuk perhitungan GLCM
    distances = [1]  # Menggunakan berbagai jarak
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 derajat
    
    # Inisialisasi array untuk menyimpan fitur untuk setiap sudut dan jarak
    asm_result = []
    kontras_result = []
    idm_result = []
    entropy_result = []
    korelasi_result = []
    
    # Menghitung GLCM dan mengekstraksi fitur
    for distance in distances:
        glcm = graycomatrix(image_resized, [distance], angles, 256, symmetric=True, normed=True)
        for angle_idx in range(len(angles)):
            asm, contrast, idm, entropy, correlation = calculate_glcm_features(glcm[:, :, 0, angle_idx])
            asm_result.append(asm)
            kontras_result.append(contrast)
            idm_result.append(idm)
            entropy_result.append(entropy)
            korelasi_result.append(correlation)
    print(entropy_result)
    return asm_result, kontras_result, idm_result, entropy_result, korelasi_result


