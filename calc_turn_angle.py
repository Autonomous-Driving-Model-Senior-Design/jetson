import glob
import os 
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt 
from support_func import * 

def turn_angle(img):
    # img = Image.open("/home/car/Desktop/WBA_160/Data/Test/Test Label/frame9.jpg")
    # img = np.expand_dims(np.array(img), axis=2)
    # print(f"1: {img.shape}") 
    # fov_boundary_bottom = fov_bottom(img) 
    fov_boundary_bottom = img.shape[0]
    fov_boundary_top, fov_top_width = fov_top(img) 
    # print(f"2: {img.shape}") 

    # normal operation 
    if fov_boundary_top != fov_boundary_bottom: 
        # compute angle based on the fov 
        fov_img = img[fov_boundary_top:fov_boundary_bottom, :, :] 
        mid_width = fov_img.shape[1]//2 
        num_black_pix_left = calc_num_black_pix(fov_img[:, :mid_width, :])
        num_black_pix_right = calc_num_black_pix(fov_img[:, mid_width:, :])
        top_left_width = find_top_left_pix_width(fov_img)
        top_right_width = find_top_right_pix_width(fov_img)

        percent_difference = abs(num_black_pix_left-num_black_pix_right) / ((num_black_pix_left+num_black_pix_right)/2) * 100
        total_num_black_pix = num_black_pix_left + num_black_pix_right 
        if total_num_black_pix < img.shape[0] * img.shape[1] / 3: # 2
            speed = 72 # 70. 72
        else:
            speed = 73 # 72
            
        # case 1; left and right have almost the same number of pixels 
        if percent_difference < 5: 
            print("go straight") 
            turn_angle = 0
            converted_angle_offset = 80
        # case 2: calculate degree of turning angle based on the leftmost pixel relative to the center
        elif num_black_pix_left > num_black_pix_right: 
            print("go left")
            turn_angle = calc_turn_angle((0, top_left_width), (fov_img.shape[0], mid_width))
            converted_angle_offset = 42 if (80 - turn_angle - 10) < 42 else (80 - turn_angle - 10)
        # case 3: calculate degree of turning angle based on the rightmost pixel relative to the center
        else: 
            print("go right")
            turn_angle = calc_turn_angle((0, top_right_width), (fov_img.shape[0], mid_width))
            converted_angle_offset = 118 if (80 + turn_angle + 10) > 118 else (80 + turn_angle + 10)

        # print(f"turn angle: {turn_angle:.2f}")
        # print(f"converted_angle_offset: {converted_angle_offset:.2f}")
        # print(f"percent_difference: {percent_difference:.4f}\n") 
        return converted_angle_offset, speed  
    else: 
        return -1, 73 # error value (skip this input in operation)


    # plt.figure()
    # plt.imshow(fov_img[:,top_left_width:,:])
    # plt.show()
    # plt.imshow(fov_img[:,:top_right_width,:])
    # plt.show()

    # print(f"\nnum_black_pix_left: {num_black_pix_left}") 
    # print(f"num_black_pix_right: {num_black_pix_right}") 
    # print(f"\ntop_left_width: {top_left_width}") 
    # print(f"top_right_width: {top_right_width}\n") 
    # print(f"percent_difference: {percent_difference}\n") 

    # print(f"\nfov_boundary_bottom: {fov_boundary_bottom}") 
    # print(f"fov_boundary_top: {fov_boundary_top}") 
    # print(f"fov_top_width: {fov_top_width}") 

    # # save each image separately 
    # fov_bottom_img = img[:fov_boundary_bottom, :, :] # pixels start at top left corner --> (height, width, 3) 
    # fov_bottom_img = Image.fromarray(fov_bottom_img) 
    # fov_bottom_img.save(result_dir+f"{i+1}"+"_fov_bottom.jpg")

    # fov_top_img = img[fov_boundary_top:, :, :] 
    # fov_top_img = Image.fromarray(fov_top_img) 
    # fov_top_img.save(result_dir+f"{i+1}"+"_fov_top.jpg")

    # fov_img = img[fov_boundary_top:fov_boundary_bottom, :, :] 
    # fov_img = Image.fromarray(fov_img) 
    # fov_img.save(result_dir+f"{i+1}"+"_fov.jpg")




# parse_image = image_list[0][:, :80, :] # pixels start at top left corner --> (height, width, 3) 
# plt.figure()
# plt.imshow(parse_image)
# plt.show() 

# print(image_list[0]) 

# # show image just to verify 
# plt.figure()
# plt.imshow(image_list[0])
# plt.show() 

