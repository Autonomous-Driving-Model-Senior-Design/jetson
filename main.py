import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np 

from wba import WeBACNN
from dataset import dataset
from train_model import train_model
import time 
import gc 

def main():
    print(torch.cuda.is_available())
    device = torch.device("cuda:0")
    model = WeBACNN(device)
    model.load_state_dict(torch.load("model_light.pt", map_location=device))
    model.to(device)
    print(device)

    with torch.no_grad():
        model.eval()

        # Test dataset
        test_data_obj = dataset(test=True)
        test_data_loader = DataLoader(test_data_obj, batch_size=1, shuffle=False)

        result_path = "result/" # Fill in the path
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # Do not need to test label (only predict on test dataset)
        for data, _, img_name in test_data_loader:
            data = data.to(device)
            print("enter inference")
            # for _ in range(5000): 
            start = time.time() 
            predictions, _, _ = model(data)
            print(f"elpsed time: {time.time() - start}")
            print("done with prediction")

            # Display prediction result
            for i in range(len(predictions)):
                pred = predictions[i].permute(1,2,0).cpu().detach().numpy()
                print(f"pred: {pred}")
                
                thresh = 0.5 # Post-processing threshold
                thresh_pred = pred
                thresh_pred[thresh_pred < thresh] = 0 # post-processing, value less than threshold is set to 0
                thresh_pred[thresh_pred > (thresh)] = 1 # post-processing, value greater than threshold is set to 1
                print(i, thresh_pred.shape)

                # test_img=thresh_pred.permute(1,2,0).cpu().detach().numpy()
                plt.figure()
                plt.imshow(thresh_pred, cmap="gray")
                plt.savefig(f"{result_path}{img_name[0]}")  
                plt.close()
                #im = Image.fromarray((thresh_pred * 255).astype(np.uint8))
                #m.save(f"{os.getcwd()}{result_path}{img_name}.png")  
                
                del pred 
                del thresh 
                del thresh_pred 
                gc.collect()
                torch.cuda.empty_cache() 


            del data 
            del predictions 
            gc.collect()  
            torch.cuda.empty_cache()         

if __name__ == "__main__":
    main()
