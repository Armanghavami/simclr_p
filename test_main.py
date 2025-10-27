import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
import matplotlib.pyplot as plt

from model import model_1
from loss_nt_xent import nl_xent
from data_aug import BrainSimCLRViewGenerator, view1_transform, view2_transform


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")





full_dataset = datasets.CIFAR10(root='data', download=True, train=True)
small_dataset = torch.utils.data.Subset(full_dataset, range(16))

view_generator = BrainSimCLRViewGenerator(view1_transform, view2_transform)

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, subset, view_generator):
        self.subset = subset
        self.view_generator = view_generator

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.view_generator(img), label

dataset = SimCLRDataset(small_dataset, view_generator)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

batch_views, _ = next(iter(dataloader))
z_i_batch = torch.stack([v[0] for v in batch_views])
z_j_batch = torch.stack([v[1] for v in batch_views])

plt.figure(figsize=(8,4))
batch_size_actual = z_i_batch.shape[0]  # get the real batch size
plt.figure(figsize=(8, 4))
for i in range(batch_size_actual):
    plt.subplot(2, batch_size_actual, i+1)
    plt.imshow(z_i_batch[i].permute(1,2,0))
    plt.title("z_i")
    plt.axis("off")

    plt.subplot(2, batch_size_actual, i+1+batch_size_actual)
    plt.imshow(z_j_batch[i].permute(1,2,0))
    plt.title("z_j")
    plt.axis("off")
plt.show()


model = model_1().to(device)
optim = Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=3)


#loss_fn = nl_xent(z_i_batch, z_j_batch, temp=0.5)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_views, _ in dataloader:
        z_i = torch.stack([v[0] for v in batch_views]).to(device)
        z_j = torch.stack([v[1] for v in batch_views]).to(device)

        embeddings_i = model(z_i)
        embeddings_j = model(z_j)

        loss_value = nl_xent(embeddings_i, embeddings_j, temp=0.5)

        optim.zero_grad()
        loss_value.backward()
        optim.step()

        running_loss += loss_value.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
    scheduler.step(epoch_loss)
