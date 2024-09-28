
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# MNIST 데이터셋 로드 및 변환
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer 1
        # 입력 채널: 1, 출력 채널: 16, 커널 크기: 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # FC1 : 중간 계층 (입력 = 16 * 14 * 14, 출력 = 1024)
        self.fc1 = nn.Linear(16 * 14 * 14, 1024)

        # Output : 출력 계층 (입력 = 1024, 출력 = 10 : 클래스의 수)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # Convolution layer 후 활성화 함수 : relu
        x = F.relu(self.conv1(x))
        # Max pooling : 2x2
        x = F.max_pool2d(x, 2)

        # feature map을 1차원 벡터로 변환 (batch 크기 유지)
        x = x.view(-1, 16 * 14 * 14)

        # FC1 : fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # FC2 : Output layer
        x = self.fc2(x)

        return x

# 모델 초기화, 손실 함수 및 옵티마이저 설정
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
#optimizer = optim.Adam(model.parameters(), lr = 0.001)

# 학습
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: %d    Loss: %6f' %(epoch+1, loss.item()))
    #print(f'Train Epoch: {epoch}  Loss: {loss.item():.6f}')


# 테스트
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # 배치 손실 합
        test_loss += criterion(output, target).item()
        # 가장 높은 확률 가진 클래스 선택
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
print("\nTest set: Accuracy: %d/%d (%d%%)\n" % (correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
#print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')