import certifi
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST 데이터셋 로드 및 변환

# ? transforms.Compose는 여러 개의 변환을 순차적으로 적용합니다.
transform = transforms.Compose([
    transforms.ToTensor(), # ? 이미지를 PyTorch 텐서로 변환합니다.
    transforms.Normalize((0.1307,), (0.3081,)) # ? MNIST 데이터셋의 평균과 표준편차를 사용하여 이미지를 정규화합니다. (0.1307,)는 MNIST 데이터의 평균, (0.3081,)는 표준편차입니다. 각 채널에 대해 해당 값을 사용해 정규화합니다.

])

# ? MNIST 데이터셋을 다운로드하고 불러옵니다. train=True는 훈련 데이터를, train=False는 테스트 데이터를 의미합니다. **transform**은 데이터에 사전 정의된 변환을 적용합니다.
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# ? 테스트 데이터는 batch_size=1000으로 설정하여 한 번에 1000개의 데이터를 로드합니다.

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


# ? : SimpleCNN 클래스는 CNN 모델을 정의합니다. nn.Module을 상속받아 __init__() 함수에서 레이어를 정의하고, forward() 함수에서 데이터가 어떻게 처리되는지 기술합니다.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer 1
        # 입력 채널: 1, 출력 채널: 16, 커널 크기: 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 중간 계층
        self.fc1 = nn.Linear(32 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        # Output : 출력 계층
        self.fc3 = nn.Linear(512, 10)

    # ? 모델의 순전파 과정이 기술된 함수입니다. 입력 이미지가 CNN 레이어와 FC 레이어를 어떻게 통과하는지를 나타냅니다.
    def forward(self, x):

        x = F.leaky_relu(self.conv1(x)) # 첫 번째 합성곱 레이어에 입력 이미지를 넣고 Leaky ReLU 활성화 함수 적용.
        x = F.max_pool2d(x, 2) # 2x2 크기의 max pooling을 적용하여 이미지 크기를 줄입니다 (28x28 → 14x14).
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
       
        x = x.view(-1, 32 * 7 * 7) #  2D feature map을 1D로 변환하여 FC 레이어에 넣기 위해 펼칩니다.
        # ully connected layer with leaky ReLU
        x = F.leaky_relu(self.fc1(x)) 
        x = F.leaky_relu(self.fc2(x)) 

        # Output layer
        x = self.fc3(x) # 두 번째 fully connected 레이어로, 10개의 클래스에 대한 예측을 출력합니다.

        return x


model = SimpleCNN().to(device) 
criterion = nn.CrossEntropyLoss() 
# optimizer = optim.SGD(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(10):
    model.train() # 모델을 훈련 모드로 설정합니다. 이는 드롭아웃(dropout)이나 배치 정규화(batch normalization)와 같은 특정 레이어들이 학습 중에만 활성화되도록 하기 위한 것입니다.
    for batch_idx, (data, target) in enumerate(train_loader): # 미니배치 단위로 데이터를 불러옵니다.
        data, target = data.to(device), target.to(device) # 데이터와 타겟을 GPU 혹은 CPU로 이동합니다.
        optimizer.zero_grad() # 기울기를 초기화합니다.
        output = model(data) # 데이터를 모델에 입력하고 출력을 계산합니다.
        loss = criterion(output, target) # 출력과 타겟 사이의 손실을 계산합니다.
        loss.backward() # 손실에 대한 기울기를 계산합니다.
        optimizer.step() # 계산된 기울기를 사용해 모델의 가중치를 업데이트합니다.

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