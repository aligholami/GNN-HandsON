from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.object_detector import ObjectDetector
from models.interaction_network import InteractionNetwork
from detectron2.config import get_cfg


def train(args, models, device, train_loader, optimizer, epoch):
    object_detector = models['object_detector']
    interaction_network = models['interaction_network']

    interaction_network.train()
    object_detector.eval()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        region_feature_matrix, batch_indexes = object_detector([data])
        output = interaction_network(region_feature_matrix, batch_indexes)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, models, device, test_loader):
    object_detector = models['object_detector']
    interaction_network = models['interaction_network']
    object_detector.eval()
    interaction_network.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            regions = object_detector(data)
            output = interaction_network(regions)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='dataset to run the experiment with (Default: MNIST). You can use CIFAR10 as well.')
    parser.add_argument('--dataset-metadata', type=str, default='../data/cifar/cifar-10-batches-py/batches.meta')
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
                        default='./region-proposal/detectron2/configs/Base-RCNN-FPN.yaml',
                        help='Path to a config file for region proposal network.')
    parser.add_argument('--rpn-pre-trained-file', type=str,
                        default='./region-proposal/detectron2/ImageNetPretrained/FAIR/model_final.pkl')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'MNIST':
        train_dataset = datasets.MNIST('../data/mnist', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))

        val_dataset = datasets.MNIST('../data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    elif args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))

        val_dataset = datasets.CIFAR10('../data/cifar', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    else:
        print(f"{args.dataset} is an invalid dataset.")
        exit(0)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    model_config = {
        'D_S': 1,
        'D_R': 1,
        'D_E': 20,
        'D_X': 50176,
        'D_P': 2048,
        'NUM_CLASSES': 10,
        'num_max_regions': 40
    }

    # Get dataset information for CIFAR 10
    cfg = get_cfg()
    cfg.merge_from_file(args.rpn_cfg_path)
    cfg.MODEL.WEIGHTS = args.rpn_pre_trained_file
    object_detector = ObjectDetector(cfg, model_config['num_max_regions'], device).to(device)
    interaction_network = InteractionNetwork(model_config).to(device)
    optimizer = optim.Adam(list(interaction_network.parameters()) + list(object_detector.parameters()), lr=args.lr)

    models = {
        'interaction_network': interaction_network,
        'object_detector': object_detector
    }

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, models, device, train_loader, optimizer, epoch)
        test(args, models, device, val_loader)
        scheduler.step()

    if args.save_model:
        for name, model in models.items():
            torch.save(model.state_dict(), name + '.pt')


if __name__ == '__main__':
    main()
