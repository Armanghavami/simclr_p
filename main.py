from model import model_1
from loss_nt_xent import nl_xent
from torch.utils.data import random_split
import torch.nn as nn 
import torch
import torchvision
from tqdm import tqdm
# see the problem of dif sizes 
# 1

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config
import torch 
from data_aug import SimCLRViewGenerator ,view1_transform, view2_transform

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision

# for mac only 
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available — using CPU instead")




# the data augmentations , load the data 
#return [self.view1(x), self.view2(x)



view_generator = SimCLRViewGenerator(view1_transform, view2_transform)

dataset = torchvision.datasets.CIFAR10("data.pth", train=True, download=True,transform=view_generator)
small_dataset, _ = random_split(dataset, [int(len(dataset)*config["size_dataset_train"]), int(len(dataset) - len(dataset)*config["size_dataset_train"])])
dataloader = DataLoader(small_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=4,pin_memory=True,prefetch_factor=2)


# Initializing the model , obtim , loss 
model = model_1().to(device)


checkpoint = torch.load("simclr_full.pth", map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.train()







#optim = Adam(model.parameters(), lr=config['learning_rate']) # change with something with momentom 

optim = torch.optim.SGD(
    model.parameters(),
    lr=config['learning_rate'],
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)


scheduler = ReduceLROnPlateau(optim, mode='min', factor=config['lr_factor'], patience=config['lr_patioence'])
num_epochs=config['num_epochs']

for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    running_loss = 0.0



    for (view1, view2), _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):


        view1, view2 = view1.to(device), view2.to(device)
        z_i = model(view1)
        z_j = model(view2)
        loss_value = nl_xent(z_i, z_j, temp=0.5)


        optim.zero_grad()
        loss_value.backward()
        optim.step()

        running_loss += loss_value.item()


    epoch_loss = running_loss / len(dataloader)
    current_lr = optim.param_groups[0]['lr']

    print(f"Epoch [{epoch+1}/{config['num_epochs']}] Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")


    # Step scheduler
    scheduler.step(epoch_loss)

    # Save model every 10 epochs 
    if (epoch + 1) % 10 == 0:
        encoder = nn.Sequential(*list(model.resnet.children())[:-1])
        torch.save(model.state_dict(), "simclr_full.pth")
        torch.save(encoder.state_dict(), "simclr_encoder.pth")
        print(f"✅ Model saved at epoch {epoch+1}")



encoder = nn.Sequential(*list(model.resnet.children())[:-1])
torch.save(encoder.state_dict(), "simclr_encoder.pth")
print("Encoder saved successfully!")


torch.save({
    'model_state': model.state_dict(),   # full model: encoder + projection head
}, "simclr_full.pth")

print("simclr_full saved successfully!")
