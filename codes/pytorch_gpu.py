
import torch

print(torch.cuda.is_available())
print("available device: ", torch.cuda.device_count())
current_device_id = torch.cuda.current_device()
print('the current device is : ', current_device_id)
print(torch.cuda.device(current_device_id))
print(torch.cuda.get_device_name(current_device_id))

print('check loading tensor to gpu : ')
torch.ones((1)).to('cuda')
print('success')
