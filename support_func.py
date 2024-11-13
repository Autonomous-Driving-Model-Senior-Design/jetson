import numpy as np 

def fov_bottom(img): 
    '''
    @input: image in numpy \n
    @return: bottom y value of FOV (counting from top left corner of image) \n
    Note: assume FOV below 1/2 of image 
    '''
    boundary_height = img.shape[0] # set boundary to the bottom of the image 
    img_height = img.shape[0] 
    img_width = img.shape[1] 

    # start testing the image from 1/2 of the image (assume intersection of FOV starts below 1/2 of image)
    for height, row in enumerate(img[img_height//2:,:,:]): 
        num_white_pix = 0 
        for pix in row: 
        #     print(f"pix: {pix}")  

            # if the pixel is white (all three values in pix are the same) 
            if pix[0] == 255: 
                num_white_pix += 1
        
        # if the number of white pixels is at least 1/3 of the image width 
        if num_white_pix >= img_width // 3: 
            boundary_height = height + img_height//2 # img_height//2 = starting point height  
            break 
    
    return boundary_height


def fov_top(img):
	result = np.where(img == 0)

	if(result[0].size > 0):
		return result[0][0], result[1][0]
	else:
		return img.shape[0], img.shape[1]
#     '''
#     @input: image in numpy \n
#     @return: top y value for FOV \n
#     '''
#     for height, row in enumerate(img): 
#         for width, pix in enumerate(row): 
#             # break if any pixel in the row is black (detected mask)  
#             if pix[0] == 0: 
#                 return height, width
#     return img.shape[0], img.shape[1] 

    
def calc_num_black_pix(img): 
    return img.size - np.sum(img)

def find_top_left_pix_width(img):
#     print(img.shape) 
    row = img[0] # first row 
    for width, pix in enumerate(row): 
        if pix[0] == 0: return width 

def find_top_right_pix_width(img): 
    row = img[0] # first row 
    row = np.flip(row) 
    for width, pix in enumerate(row): 
        if pix[0] == 0: return len(row)-width # because row is reversed 

def calc_turn_angle(top_pix, center_pix): 
    '''
    calculate the turning angle from center_pix to top_pix \n
    @input: pix -> height, width (relative to top left corner) \n
    ''' 
    return np.arctan(abs(top_pix[1]-center_pix[1])/abs(top_pix[0]-center_pix[0])) * 180/np.pi 

def map_range(value, old_min, old_max, new_min, new_max):
    # Ensure old_min != old_max to avoid division by zero
    if old_min == old_max:
        raise ValueError("Old min and old max cannot be the same value.")
    
    # Map value from the old range to the new range
    new_value = (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return new_value



    
