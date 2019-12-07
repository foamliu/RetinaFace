import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from config import device, num_classes, training_dataset, rgb_mean, grad_clip, print_freq, num_workers
from retinaface.data import WiderFaceDetection, detection_collate, preproc, cfg_mnet
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.layers.modules import MultiBoxLoss
from retinaface.models.retinaface import RetinaFace
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger

warnings.simplefilter(action='ignore', category=FutureWarning)


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = 0
    writer = SummaryWriter()
    epochs_since_improvement = 0

    cfg = cfg_mnet
    img_dim = cfg['image_size']
    num_gpu = cfg['ngpu']
    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    gpu_train = cfg['gpu_train']

    # Initialize / load checkpoint
    if checkpoint is None:
        net = RetinaFace(cfg=cfg)
        print("Printing net...")
        print(net)
        net = nn.DataParallel(net)

        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        net = checkpoint['net']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    net = net.to(device)

    cudnn.benchmark = True

    # Loss function
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    # Custom dataloaders
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers,
                                               collate_fn=detection_collate)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
    # scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                                            net=net,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            cfg=cfg,
                                            priors=priors,
                                            epoch=epoch,
                                            logger=logger)

        writer.add_scalar('model/train_loss', train_loss, epoch)

        lr = optimizer.param_groups[0]['lr']
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model/learning_rate', lr, epoch)

        # One epoch's validation
        val_acc, thres = test(net)
        writer.add_scalar('model/valid_accuracy', val_acc, epoch)
        writer.add_scalar('model/valid_threshold', thres, epoch)

        scheduler.step(epoch)

        # Check if there was an improvement
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, net, optimizer, best_acc, is_best)


def train(train_loader, net, criterion, optimizer, cfg, priors, epoch, logger):
    net.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (images, targets) in enumerate(train_loader):
        # Move to GPU, if available
        images = images.to(device)
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                      loss=losses))

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
