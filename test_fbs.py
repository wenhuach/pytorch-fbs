from cifarnet import CifarNet, _Graph
import torch
import argparse
import math
from torchvision import datasets, transforms
import misc
import os

print = misc.logger.info
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='cifarnet')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--ratio', default=0.5, type=float)

args = parser.parse_args()
args.logdir = 'test-fbs/%s-%s-ratio-%.2f/' % (args.arch, args.dataset, args.ratio)

misc.prepare_logging(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

print('Loading checkpoint...')
model = CifarNet()
model.load_state_dict(torch.load('cifarnet.pth'))
model.eval()
model = model.cuda()

_Graph.set_global_var('ratio', args.ratio)

class PerImageStandardization(object):
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(t.mean()).div_(max(t.std(), 1 / math.sqrt(t.numel())))

        return tensor

kwargs = {'num_workers': 1, 'pin_memory': True}
if args.dataset == 'cifar10':
    test_data_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            PerImageStandardization(),
                        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

labels = []
all_preds = []

for data, target in test_data_loader:
    with torch.no_grad():
        data = data.cuda()
        labels.append(target)

        output = model.gated_forward(data)
        pred = output.max(1)[1]
        all_preds.append(pred)

acc = (torch.cat(all_preds).cpu() == torch.cat(labels)).float().mean()
print('... Test acc = %.4f' % (acc.item()))
