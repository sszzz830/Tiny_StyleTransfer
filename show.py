from torchvision import transforms
from PIL import Image
import torch
import re
from transformer_net import TransformerNet
from torchvision.transforms import ToPILImage
import utils
from thop import profile
import time

model = TransformerNet()
state_dict = torch.load('taffy.model')
model.load_state_dict(state_dict)
model.eval()
image = utils.load_image('/Users/zql/Desktop/StyleEngine/BenchMark/BenchmarkSrc.png')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
input_image = transform(image).unsqueeze(0)
with torch.no_grad():
    t1=time.time()
    output_image = model(input_image).cpu()[0]
    t1=time.time()-t1
output_image = output_image.clamp(0, 255).div(255)
tp = ToPILImage()
result = tp(output_image)
result.show()
flops, params = profile(model, (input_image,))
print('Gflops: ', flops/1E9, '\nParams: ', int(params),'\nTime: ', t1*1E3 ,'ms')