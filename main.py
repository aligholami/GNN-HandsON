from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import yaml
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.interaction_network import InteractionNetwork
from detectron2.modeling import build_model
from detectron2.config import get_cfg


def get_batch_dict_list_from_tensors(batch_data, device):
    """
    Converts a bunch of tensors with shape (batch_size, width, height, num_channels) to a list of dictionaries
    compatible with Detectron input format (https://detectron2.readthedocs.io/tutorials/models.html).
    :param dat: Batch_tensor.
    :return: list of dictionaries.
    """
    img_dict = {
        "image": None,
        "height": None,
        "width": None
    }
    img_list = []

    for img in batch_data.tolist():
        img_tensor = torch.tensor(img)
        img_dict['image'] = img_tensor.to(device)
        img_dict['height'], img_dict['width'] = img_tensor.shape[:2]
        img_list.append(img_dict)

    return img_list

def train(args, models, device, train_loader, optimizer, epoch):
    region_proposal = models['region_proposal']
    interaction_network = models['interaction_network']

    region_proposal.eval()
    interaction_network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        img_list_dict = get_batch_dict_list_from_tensors(data, device)
        regions = region_proposal(img_list_dict)
        print("RPN Output: ", regions)
        exit(0)
        output = interaction_network(regions)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, models, device, test_loader):
    region_proposal = models['region_proposal']
    interaction_network = models['interaction_network']

    region_proposal.train()
    interaction_network.train()

    region_proposal.eval()
    interaction_network.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            regions = region_proposal(data)
            output = interaction_network(regions)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def load_pre_trained_weights(path_to_pkl):
    """
    Reads a pickle file (designed for official pkl files for the detectron2 repository).
    :param path_to_pkl: A string, path to the desired pickle file.
    :return: weights, a dictionary showing the layer name as key and the numpy nd-arrays as values for the weights.
    """
    with open(path_to_pkl, 'rb') as f:
        obj = f.read()

    weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1')['model'].items()}

    return weights


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='dataset to run the experiment with (Default: MNIST). You can use CIFAR10 as well.')
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
    parser.add_argument('--rpn_cfg_path', type=str,
                        default='./region-proposal/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
                        help='Path to a config file for region proposal network.')
    parser.add_argument('--rpn_pre_trained_file', type=str,
                        default='./region-proposal/pre-trained-proposals/model_final_68b088.pkl')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'MNIST':
        train_dataset = datasets.MNIST('../data/mnist', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
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
        'D_X': 128,
        'D_P': 2048,
        'NUM_CLASSES': 10
    }

    pre_trained_rpn_weights = load_pre_trained_weights(args.rpn_pre_trained_file)
    print("Load the weights dictionary.")
    cfg = get_cfg()
    cfg.merge_from_file(args.rpn_cfg_path)
    region_proposal = build_model(cfg)
    print("Built the region proposal.")
    region_proposal.load_state_dict(pre_trained_rpn_weights)
    print("Loaded model weights.")
    interaction_network = InteractionNetwork(model_config).to(device)
    optimizer = optim.Adam(list(region_proposal.parameters()) + list(interaction_network.parameters()), lr=args.lr)

    models = {
        'region_proposal': region_proposal,
        'interaction_network': interaction_network
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
