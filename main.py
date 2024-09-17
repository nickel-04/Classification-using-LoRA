import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import litdata
import time

# Define a function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine tuning ImageWoof Dataset")
    parser.add_argument("--datapath", type=str, default="/projects/ec232/data/", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224], help="Image size")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA")

    args = parser.parse_args()
    return args

# Define a function to load data
def load_data(datapath, img_size, batch_size):
    postprocess = (
        T.Compose([
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.Resize(img_size, antialias=None)
        ]),
        nn.Identity(),
    )

    traindata = litdata.LITDataset('ImageWoof', datapath).map_tuple(*postprocess)
    valdata = litdata.LITDataset('ImageWoof', datapath, train=False).map_tuple(*postprocess)

    train_dataloader = DataLoader(traindata, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(valdata, shuffle=False, batch_size=batch_size)

    return train_dataloader, val_dataloader

# Define a function to save the model
def save_best_model(model, optimizer, epoch, loss, accuracy, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)

# Define a function to train the model
def train_model(model, train_dataloader, num_epochs, loss_criterion, optimizer):
    model.train()
    start_time = time.time()
    best_loss = float('inf') 
    for epoch in range(num_epochs):
        train_loss, total_samples, train_acc = 0.0, 0.0, 0.0

        train_bar = tqdm(iterable=train_dataloader)
        for i, batch in enumerate(train_bar):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_criterion(y_hat, y)

            train_loss += loss.item()
            total_samples += len(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_hat)
            train_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            train_bar.set_postfix(loss=train_loss / total_samples)

        avg_loss = train_loss / total_samples
        avg_accuracy = train_acc / total_samples * 100

        print(f'Train loss: {avg_loss:.4f}')
        print(f'Train accuracy: {avg_accuracy:.2f}%')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_accuracy = avg_accuracy
            save_best_model(model, optimizer, epoch, loss, best_accuracy, "model_imagewoof")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

# Define a function to test the model
def test_model(model, test_loader, loss_criterion):
    model.eval()
    with torch.no_grad():
        test_loss, correct_pred, total_samples = 0.0, 0.0, 0.0

        test_bar = tqdm(iterable=test_loader)
        for i, batch in enumerate(test_bar):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_criterion(y_hat, y)

            test_loss += loss.detach().cpu().item()
            correct_pred += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total_samples += len(x)

            test_bar.set_description(f'Test loop')
            test_bar.set_postfix(loss=test_loss / total_samples)
    print(f'Test loss: {test_loss / total_samples:.4f}')
    print(f'Test accuracy: {correct_pred / total_samples * 100:.2f}%')

class LoRAWrapper(nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        assert isinstance(linear, nn.Linear)

        self.register_buffer('orig_weight', linear.weight.data.cpu())
        if linear.bias is not None:
            self.register_buffer('orig_bias', linear.bias.data.cpu())
        else:
            self.register_buffer('orig_bias', None)

        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        self.bias = linear.bias is not None
        self.rank = rank

        self.lora_linear = nn.Sequential(
            nn.Linear(self.in_dim, rank, bias=False),
            nn.Linear(rank, self.out_dim, bias=self.bias)
        ).to(device)

        nn.init.normal_(self.lora_linear[0].weight.data, mean=0, std=0.01)

        if self.bias:
            nn.init.zeros_(self.lora_linear[1].weight.data)
            nn.init.zeros_(self.lora_linear[1].bias.data)

    def forward(self, x):
        W_0 = torch.matmul(x.to(device), self.orig_weight.t().to(device))

        if self.bias:
            W_0 += self.orig_bias.to(device)

        A_x = self.lora_linear[0](x)
        BA_x = self.lora_linear[1](A_x)

        output = W_0 + BA_x

        return output

def get_pretrained_model(num_classes, device, lora=False):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes).to(device)
    if lora is True:
        for name, param in model.named_parameters():
            if name in ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for block in model.blocks:
            block.attn.qkv = LoRAWrapper(block.attn.qkv, rank=12)
            block.attn.proj = LoRAWrapper(block.attn.proj, rank=12)
            block.attn.requires_grad_(True)
            block.ls1.requires_grad_(True)
            block.ls2.requires_grad_(True)

    return model

if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, val_dataloader = load_data(args.datapath, args.img_size, args.batch_size)

    best_loss = float("inf")
    best_accuracy = 0.0

    model = get_pretrained_model(args.num_classes, device, lora=args.lora)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print("Training with LoRA:", args.lora)
    train_model(model, train_dataloader, args.num_epochs, nn.CrossEntropyLoss(), optimizer)
    test_model(model, val_dataloader, nn.CrossEntropyLoss())
    if args.lora:
        model_save_name = 'lora_model.pth'
    else:
        model_save_name = 'full_model.pth'

    # Save the model
    torch.save(model.state_dict(), model_save_name)
