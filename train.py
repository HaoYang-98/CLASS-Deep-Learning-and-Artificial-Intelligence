import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from model.model_AlexNet import AlexNet, F_Conv_AlexNet
from model.model_LeNet import LeNet, F_Conv_LeNet
from prepare_MNIST_data import prepare_MNIST_data, MNISTDataset
import argparse

import matplotlib.pyplot as plt

def train_and_evaluate(run_id, result_dir, train_loader, validate_loader, test_loader, device, val_dataset, test_dataset):
    
    # net = LeNet()
    # net = F_Conv_LeNet()

    # net = AlexNet(num_classes=10, init_weights=True)
    net = F_Conv_AlexNet(num_classes=10, init_weights=True)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.01 if epoch < 30 else (0.01 * (0.5 ** ((epoch - 30) // 30))) if epoch < 150 else 0.0002)

    epochs = 200

    save_path = os.path.join(result_dir, f'LeNet_MNIST_{run_id}.pth')
    best_acc = 0.0

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_description("Train Epoch[{}/{}] Loss: {:.3f}".format(epoch + 1, epochs, loss))

        train_losses.append(running_loss / len(train_loader))
        # scheduler.step()

        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / len(val_dataset)
        val_accuracies.append(val_accurate)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / len(train_loader), val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(train_loader),
            }, save_path)
    print('Finished Training')

    # 绘制训练损失和验证准确率曲线并保存
    plt.figure(figsize=(10, 5))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    # 绘制验证准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.savefig(os.path.join(result_dir, f'training_loss_and_validation_accuracy_{run_id}.png'))

    print('Starting Testing')
    # 测试
    net.eval()
    checkpoint = torch.load(save_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    acc = 0.0
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            outputs = net(test_images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels).sum().item()
            correct = torch.eq(predict_y, test_labels).squeeze()
            for i in range(len(test_labels)):
                label = test_labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    test_accurate = acc / len(test_dataset)
    print('Test Accuracy: {:.3f} (Epoch: {})'.format(test_accurate, epoch+1))

    # 打印每个类别的准确率
    for i in range(10):
        if class_total[i] > 0:
            print('Accuracy of class {}: {:.3f}'.format(i, class_correct[i] / class_total[i]))

    # 保存测试结果到txt文件
    with open(os.path.join(result_dir, 'test_results.txt'), 'a') as f:
        f.write(f'Run {run_id}: Test Accuracy: {test_accurate:.3f} (Epoch: {epoch+1})\n')
        for i in range(10):
            if class_total[i] > 0:
                f.write('Accuracy of class {}: {:.3f}\n'.format(i, class_correct[i] / class_total[i]))

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate LeNet on MNIST dataset.')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number (default: 0)')
    parser.add_argument('--expriment', type=int, default=0, help='Experiment type for MNIST data preparation (default: default)')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # 预处理 MNIST 旋转数据
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = prepare_MNIST_data(True, expriment=args.expriment)

    # **调试信息**
    print(f"Train Data Shape: {train_data.shape}")  # 应该是 [275000, 1, 28, 28]
    print(f"Validation Data Shape: {validation_data.shape}")  # 应该是 [5000, 1, 28, 28]
    print(f"Test Data Shape: {test_data.shape}")  # 应该是 [10000, 1, 28, 28]
    
    # 创建 PyTorch Dataset
    train_dataset = MNISTDataset(train_data, train_labels)  
    val_dataset = MNISTDataset(validation_data, validation_labels)  
    test_dataset = MNISTDataset(test_data, test_labels)  

    # 创建 DataLoader
    batch_size = 4096
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validate_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)

    print("Using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))

    # 创建结果文件夹
    current_time = datetime.now().strftime("%m-%d-%H-%M-%S")
    result_dir = os.path.join('./result_AlexNet_F_Conv', current_time)
    os.makedirs(result_dir, exist_ok=True)

    for run_id in range(1, 6):
        print(f"Starting run {run_id}")
        train_and_evaluate(run_id, result_dir, train_loader, validate_loader, test_loader, device, val_dataset, test_dataset)

if __name__ == '__main__':
    main()
