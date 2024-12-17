import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import ssl

context = ssl.create_default_context()

def load_data(dataset, train_size, test_size, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset == 'CIFAR10':
        data_dir  = "././././data"
        trainset = torchvision.datasets.CIFAR10(root=data_dir , train=True, download=False , transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    elif dataset == 'CIFAR100':
        data_dir  = "././././data"
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == 'ImageNet':
        traindir = '/path/to/imagenet/train'
        testdir = '/path/to/imagenet/val'
        trainset = torchvision.datasets.ImageNet(traindir, split='train', transform=transform)
        testset = torchvision.datasets.ImageNet(testdir, split='val', transform=transform)
    else:
        raise ValueError("Dataset not supported")

    trainset, _ = torch.utils.data.random_split(trainset, [train_size, len(trainset) - train_size])
    testset, _ = torch.utils.data.random_split(testset, [test_size, len(testset) - test_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def train_model(trainloader, model, criterion, optimizer, writer, max_training_time=300, epochs=5):
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            print("In train, one passe en data")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Enregistrer la perte à chaque étape
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(trainloader) + i)
            # Enregistrer les histogrammes des sorties du modèle à chaque étape
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().detach().cpu().numpy(), epoch * len(trainloader) + i)
            if time.time() - start_time > max_training_time:
                print('Training time exceeded maximum time limit of 5 minutes.')
                return  # Exit training loop if max training time is exceeded
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
        # Enregistrer la perte moyenne à chaque époque
        writer.add_scalar('Loss/Train_Epoch', running_loss / len(trainloader), epoch + 1)
    print('Finished Training')
    return model

def evaluate_model(testloader, model, writer, metrics=["accuracy"]):
    metrics_values = {}
    for metric in metrics:
        if metric == "accuracy":
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            metrics_values[metric] = accuracy
            writer.add_scalar('Accuracy/Test', accuracy)
            print('Accuracy of the network on the test images: %d %%' % accuracy)
        # Autres métriques à calculer
        else:
            raise ValueError("Unsupported metric: {}".format(metric))
            # Calcul de la métrique
            # metric_value = calculate_metrics(outputs, labels)
            # metrics_values[metric] = metric_value
            # writer.add_scalar('{}/Test'.format(metric.capitalize()), metric_value)

    return metrics_values  # Retourne toutes les métriques calculées


# Example usage:
# trainloader, testloader = load_data('CIFAR10', train_size=5000, test_size=1000, batch_size=64)
# model = CNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# train_model(trainloader, testloader, model, criterion, optimizer, epochs=2, metrics=["accuracy", "precision", "recall"])
