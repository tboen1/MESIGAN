from skimage import io as img
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from sklearn.feature_extraction import image
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from skimage.filters import gaussian
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from skimage.color import rgb2gray
import anchoring
import torch

def generate_matrix(patches):
  matrix = []
  for i in range(9):
    for j in range(9):
      vec = patches[i][j][0].ravel()
      #mean_shifted = vec
      mean_shifted = (vec-np.mean(vec))
      mean_shifted_norm_vec = mean_shifted/np.linalg.norm(mean_shifted)
      #mean_shifted_normed_vec = vec/np.linalg.norm(vec)
      matrix.append(mean_shifted_norm_vec)
  matrix = np.array(matrix)
  return matrix 

def index2patch(index): #HARDCODED
  #index = index + 1 
  j = int(index%9)
  i = int((index-j)/9)
  return i,j

def patch_match(target_patches, match):
  source_patches = np.zeros([9,9,1,200,200,3])
  for index in range(len(match)):
    source_i, source_j = index2patch(index)
    target_i, target_j = index2patch(match[index])

    source_patches[source_i][source_j][0] = target_patches[target_i][target_j][0]
  return source_patches

def L2(patch1, patch2):
    
    vec1 = patch1.ravel()
    vec2 = patch2.ravel()
    vec1_ms = vec1 - np.mean(vec1)
    vec2_ms = vec2 - np.mean(vec2)
    vec1_ms_norm = vec1_ms/np.linalg.norm(vec1_ms)
    vec2_ms_norm = vec2_ms/np.linalg.norm(vec2_ms)

    return np.dot(vec1_ms_norm, vec2_ms_norm)

def convert(im):
  im = (im*255).astype(np.uint8)
  return im

def regroup(source_patches, target_shape, overlap):
  target_image = np.zeros(target_shape)
  height, width, channel = target_shape
  x0, y0 = [0,0]
  rows, columns, _, x_size, y_size, _ = source_patches.shape
  for i in range(rows):
    for j in range(columns):
      target_image[y0:y0 + y_size,x0:x0 + x_size,:] = source_patches[i][j][0]
      x0 = x0 + overlap
    y0 = y0 + overlap
    x0 = 0
  return target_image

def new_mask(name, pixels, alpha = 0):
    x = np.linspace(0,1,pixels)
    y = np.zeros(pixels)
    for i in range(pixels):
      y[i] = function(alpha,0,x[i])

    map = np.ones([200,200,3])
    for i in range(pixels):
      map[:,i,:]*=(y[i])
    map[:,pixels:,:] = 0
    #map = 1-map

    if name == 'L':
        return map
    if name == 'T':
        return np.rot90(np.rot90(np.rot90(map)))

    if name == 'L_shape':
        map3 = np.zeros([200,200,3])
        for i in range(pixels):
          map3[0:200-i,0:200-i,:] = (1-y[i])
        map3 = 1-map3
        return np.rot90(np.rot90(map3))

def anchor(name, im1, im2 = None):
  anchor_im = np.ones([200,200,3])
  if name == 'L':
    anchor_im = np.ones([200,200,3])
    anchor_im[:,:100,:] = im1[:,100:,:]
  if name == 'T': #fix this 
    anchor_im = np.ones([200,200,3])
    anchor_im[:100,:,:] = im1[100:,:,:]
  if name == 'L_shape': ##NOT TRUE
    anchor_im = np.ones([200,200,3])
    anchor_im[:,:100,:] = im1[:,100:,:]
    anchor_im[:100,:,:] = im2[100:,:,:]
  return anchor_im

def function(alpha, x0, x):
    y = 1-((1-alpha)*(x-x0)+alpha*x**2)
    return max(min(y, 1), 0) #clamping

def threshold(anchor_image, window_size=25):
  avg = np.zeros(anchor_image.shape)
  avg = avg[:,:,0]
  for i in range(3):
    image = anchor_image[:,:,i]
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_sauvola = image > thresh_sauvola
    avg = avg + (binary_sauvola)
  avg = avg/3
  avg_new = np.stack((avg,)*3, axis=-1)
  return avg_new

def get_new_direction(name, anchor1, window_size = 25, alpha = 0, pixels = 100, anchor2=None):
    filtered1 = threshold(anchor1, window_size = window_size)
    if anchor2 is None:
        direction = (1-anchor(name, filtered1))*new_mask(name, pixels, alpha = alpha)
    if anchor2 is not None:
        filtered2 = threshold(anchor2, window_size = window_size)
        direction = anchor(name, filtered1, filtered2)*new_mask(name, pixels, alpha = alpha)
    return direction

def generate_image(model_map2, fake_patches, input_name, window_size = 3, alpha = 0, pixels = 100, insert_limit = 1):

    anchor_patches = np.empty([fake_patches.shape[0],fake_patches.shape[1],1,200,200,3])

    for i in range(fake_patches.shape[0]):
      for j in range(fake_patches.shape[1]):
        insert_limit = 1
        window_size = 1
        alpha = 0
        pixels = 100
        sigma = 10
        factor = 1
        model_name = int(model_map2[i][j])
        base = fake_patches[i][j][0]
        if i == 0 and j == 0:
            im = anchoring.generate('{}_basis_{}.jpg'.format(input_name, model_name))
        
        elif i == 0:
            name = 'L'
            prev_im = anchor_patches[0][j-1][0]
            new_direction = get_new_direction(name, prev_im,
                                              window_size = window_size, alpha = alpha, pixels = pixels)
            im = anchoring.generate('{}_basis_{}.jpg'.format(input_name, model_name),
                                anchor_image=convert(anchor(name, prev_im)), 
                              direction = convert(new_direction),
                              base = base, 
                              insert_limit = insert_limit,
                              factor = factor)
        elif j == 0:
            name = 'T'
            prev_im = anchor_patches[i-1][j][0]
            new_direction = get_new_direction(name, prev_im,
                                              window_size = window_size, alpha = alpha, pixels = pixels)
            im = anchoring.generate('{}_basis_{}.jpg'.format(input_name, model_name),
                        anchor_image=convert(anchor(name, prev_im)), 
                      direction = convert(new_direction),
                      base = base, 
                      insert_limit = insert_limit,
                      factor = factor)
        else:
            name = 'L_shape'
            prev_im_L = anchor_patches[i][j-1][0]
            prev_im_T = anchor_patches[i-1][j][0]
            new_direction = get_new_direction(name, prev_im_L, anchor2 = prev_im_T,
                                              window_size = window_size, alpha = alpha, pixels = pixels)
            im = anchoring.generate('{}_basis_{}.jpg'.format(input_name, model_name),
                        anchor_image=convert(anchor(name, prev_im_L, prev_im_T)), 
                      direction = convert(new_direction),
                      base = base, 
                      insert_limit = insert_limit,
                      factor = factor)
              
        anchor_patches[i][j][0] = im

    return anchor_patches

def model_assignment(basis_names, input_name, fake_patches):
    model_map = np.zeros([fake_patches.shape[0],fake_patches.shape[1]])
    for i in range(fake_patches.shape[0]):
      for j in range(fake_patches.shape[1]):
        test_image = fake_patches[i][j][0]
        MAX_ERROR = 1000 #dumb variable
        for model_number in basis_names:
            model_name = "{}_basis_{}.jpg".format(input_name, model_number)
            print(model_name)
            Noise_Solutions, reconstructed_image, REC_ERROR = anchoring.invert_model(test_image, 
                                                                              model_name, 
                                                                              scales2invert = 1,
                                                                              show = False)
            if REC_ERROR < MAX_ERROR:
              MAX_ERROR = REC_ERROR
              model_map[i][j] = model_number
    return model_map

def invert_patches(real_patches, model_name, dir):
    noise_solutions_dict = {}
    row = real_patches.shape[0]
    col = real_patches.shape[1]

    for i in range(row):
        noise_solutions_dict[i] = {}
        for j in range(col):
          test_image = real_patches[i][j][0]

          noise_solutions, reconstructed_image, REC_ERROR = anchoring.invert_model(test_image, 
                                                                                  model_name,
                                                                                  show=False)
          noise_solutions_dict[i][j] = {}
          noise_solutions_dict[i][j]['noise_solution'] = noise_solutions
          noise_solutions_dict[i][j]['MSE_LOSS'] = REC_ERROR.detach().cpu().numpy().item()
          reconstructed_patches[i][j][0] = reconstructed_image

          torch.save(noise_solutions_dict, dir.format(model_name + '_noise_solutions.pth'))
          torch.save(reconstructed_patches, dir.format(model_name + '_rec_patch.pth'))

def build_error_maps(basis_names, real_patches):
    row = real_patches.shape[0]
    col = real_patches.shape[1]
    error_map = np.zeros(row, col)
    
    all_solutions = []

    for name in basis_names:
        new_dict = torch.load(dir.format(name + '_noise_solutions.pth'))
        all_solutions.append(new_dict)

    for i in range(row):
      for j in range(col):
        REC_ERR_list = []
        for idx, name in enumerate(basis_names):
            REC_ERR_list.append(all_solutions[idx][i][j]['MSE_LOSS'])
        error_map[i][j] = min(REC_ERR_list)

    return error_map

def basis_finding(real_patches, input_name, max_num):
    basis_names = []
    count = 0
    while count < max_num:
        model_name = '{}_basis_{}'.format(input_name, count)

        if count != 0:
            invert_patches(real_patches, model_name)

            error_map = build_error_maps(basis_names, real_patches)
            basis_i, basis_j = index2patch(np.argmax(error_map))

            input_image = real_patches[basis_i][basis_j][0] #check this 
            plt.imsave(dir.format(model_name + '.jpg'),input_image)
        else:
            input_image = real_patches[0][0][0] #train on first patch 
            plt.imsave(dir.format(model_name + '.jpg'),input_image)
        
        #TRAIN A NEW MODEL

        basis_names.append(model_name)
        count = count+1
