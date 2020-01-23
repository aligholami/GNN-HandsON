from torchvision import datasets, transforms
import torch.optim as optim
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from detectron2.config import get_cfg

import sys

sys.path.insert(0, '../')
from models.object_detector import ObjectDetector


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(50176, 64)
        self.conv2 = GCNConv(64, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train(args, models, device, train_loader, optimizer, epoch):
    rf_extractor = models['rf_extractor']
    gnn = models['gnn']

    rf_extractor.eval()
    gnn.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        region_features = rf_extractor([data]).to(device)
        num_nodes = region_features.batch.shape[0]
        modified_target = []
        for node in range(num_nodes):
            graph_idx = region_features.batch[node]
            modified_target.append(target[graph_idx])
        modified_target = torch.tensor(modified_target, dtype=torch.long).to(device)
        output = gnn(region_features)
        loss = F.nll_loss(output, modified_target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, models, device, val_loader):
    rf_extractor = models['rf_extractor']
    gnn = models['gnn']

    rf_extractor.eval()
    gnn.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in val_loader:
            data, target = data.to(device), target.to(device)
            region_features = rf_extractor([data])
            output = gnn(region_features)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='dataset to run the experiment with (Default: MNIST). You can use CIFAR10 as well.')
    parser.add_argument('--dataset-metadata', type=str, default='../../data/cifar/cifar-10-batches-py/batches.meta')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--rpn-cfg-path', type=str,
                        default='../region-proposal/detectron2/configs/Base-RCNN-FPN.yaml',
                        help='Path to a config file for region proposal network.')
    parser.add_argument('--rpn-pre-trained-file', type=str,
                        default='../region-proposal/detectron2/ImageNetPretrained/FAIR/model_final.pkl')
    parser.add_argument('--train-region-features-path', type=str, default='/local-scratch/region-features/train.h5')
    parser.add_argument('--validation-region-features-path', type=str, default='../region-features/validation.h5')
    parser.add_argument('--num-max-regions', type=int, default=30)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))

    val_dataset = datasets.CIFAR10('../data/cifar', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    cfg = get_cfg()
    cfg.merge_from_file(args.rpn_cfg_path)
    cfg.MODEL.WEIGHTS = args.rpn_pre_trained_file
    rf_extractor = ObjectDetector(cfg, device).to(device)
    gnn = Net().to(device)

    models = {
        'rf_extractor': rf_extractor,
        'gnn': gnn
    }
    optimizer = optim.Adam(list(rf_extractor.parameters()) + list(gnn.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, models, device, train_loader, optimizer, epoch)
        test(args, models, device, val_loader)


if __name__ == '__main__':
    main()