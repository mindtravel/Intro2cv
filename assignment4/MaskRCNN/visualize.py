import utils
from dataset import SingleShapeDataset
from utils import plot_save_output
import torch
import torch.utils.data


dataset_test = SingleShapeDataset(10)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 

num_classes = 4
 
# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes).double()

# model.load_state_dict(torch.load(r'/media/2T/home/zxl/code/Intro2cv/assignment4/icv_hw4/intro2cv_maskrcnn_pretrained.pth', map_location='cpu'))
model.load_state_dict(torch.load(r'/media/2T/home/zxl/code/Intro2cv/assignment4/MaskRCNN/results/maskrcnn_1.pth', map_location='cpu'))

model.eval()

# print("test")

path = "results/" 
for i in range(4):
    print("i: ", i)
    
    imgs, labels = dataset_test[i]
    output = model([imgs])
    plot_save_output(path+str(i)+"_result.png", imgs, output[0])