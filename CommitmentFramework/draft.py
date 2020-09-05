import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

train_root = '../dataset-commit/train/'
test_root = '../dataset-commit/test/'
cmc_root = '../dataset-commit/cm-center/'
cme_root = '../dataset-commit/cm-edge/'

# transform函数组合
train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])


# 使用ImageFolder读取数据
train_data =  torchvision.datasets.ImageFolder(
        root=train_root,
        transform=train_transform
    )

test_data =  torchvision.datasets.ImageFolder(
        root=test_root,
        transform=train_transform
    )
cmc_data =  torchvision.datasets.ImageFolder(
        root=cmc_root,
        transform=train_transform
    )

cme_data =  torchvision.datasets.ImageFolder(
        root=cme_root,
        transform=train_transform
    )

# 定义数据加载器
train_set = torch.utils.data.DataLoader(
    train_data,
    batch_size=1,
    shuffle=True
)

test_set = torch.utils.data.DataLoader(
    test_data,
    batch_size=1,
    shuffle=True
)
cmc_set = torch.utils.data.DataLoader(
    cmc_data,
    batch_size=1,
    shuffle=True
)

cme_set = torch.utils.data.DataLoader(
    cme_data,
    batch_size=1,
    shuffle=True
)


# print(train_set)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(36992, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 36992)
        x = self.dense(x)
        return x

model = Model()
cost = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 5000


for epoch in range(n_epochs):
    running_loss = 0.0
    running_corrrect = 0
    size_train = 0
    size_test = 0
    size_cmc=0
    size_cme=0
    print("Epoch  {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for data in train_set:
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train).float()
        outputs = model(X_train)
        label=torch.zeros(1,2,dtype=torch.float)
        label[0][y_train.data.long()]=1
        optimizer.zero_grad()
        loss = cost(outputs, label)
        # print(loss)

        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        _, pred = torch.max(outputs.data, 1)
        _, label = torch.max(label.data, 1)
        # print(pred, label)
        size_train += 1
        if pred == label:
            running_corrrect += 1
            # print(running_corrrect)

    testing_correct = 0
    for data in test_set:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        size_test += 1
        if (pred == y_test.data)[0]:
            testing_correct += 1

    cmc_correct = 0
    for data in cmc_set:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        # print(outputs)
        _, pred = torch.max(outputs.data, 1)
        size_cmc += 1
        if (pred == y_test.data)[0]:
            cmc_correct += 1

    cme_correct = 0
    for data in cme_set:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        # print(outputs)
        _, pred = torch.max(outputs.data, 1)
        size_cme += 1
        if (pred == y_test.data)[0]:
            cme_correct += 1

    print("Loss is {:.4f}, Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%,CM-C Accuracy is:{:.4f}%,CM-E Accuracy is:{:.4f}%"
          .format(running_loss / size_train, 100 * running_corrrect / size_train,
                  100 * testing_correct / size_test, 100 * cmc_correct / size_cmc, 100 * cme_correct / size_cme))