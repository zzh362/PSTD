"""Train directional marking point detector."""
import math
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import config
import data
import yaml
import util
from model import DirectionalPointDetector
import numpy as np


def plot_prediction(logger, image, marking_points, prediction):
    """Plot the ground truth and prediction of a random sample in a batch."""
    rand_sample = random.randint(0, image.size(0)-1)
    sampled_image = util.tensor2im(image[rand_sample])
    logger.plot_marking_points(sampled_image, marking_points[rand_sample],
                               win_name='gt_marking_points')
    sampled_image = util.tensor2im(image[rand_sample])
    pred_points = data.get_predicted_points(prediction[rand_sample], 0.01)
    if pred_points:
        logger.plot_marking_points(sampled_image,
                                   list(list(zip(*pred_points))[1]),
                                   win_name='pred_marking_points')


def generate_objective(marking_points_batch, device, is512 = True):
    """Get regression objective and gradient for directional point detector."""
    batch_size = len(marking_points_batch)
    size = 16 if is512 else 15#30
    objective = torch.zeros(batch_size, config.NUM_FEATURE_MAP_CHANNEL,
                            size, size,
                            device=device)
    gradient = torch.zeros_like(objective)
    gradient[:, 0].fill_(1.)
    for batch_idx, marking_points in enumerate(marking_points_batch):
        for marking_point in marking_points:
            col = math.floor(marking_point.x * size)
            row = math.floor(marking_point.y * size)
            # Confidence Regression
            objective[batch_idx, 0, row, col] = 1.
            # Makring Point Shape Regression
            objective[batch_idx, 1, row, col] = marking_point.shape
            # Offset Regression
            objective[batch_idx, 2, row, col] = marking_point.x*size - col
            objective[batch_idx, 3, row, col] = marking_point.y*size - row
            # Direction Regression
            direction0 = marking_point.direction0
            objective[batch_idx, 4, row, col] = (math.cos(direction0) +1)/2
            objective[batch_idx, 5, row, col] = (math.sin(direction0) +1)/2
            direction1 = marking_point.direction1
            objective[batch_idx, 6, row, col] = math.cos(direction1)/2 +0.5
            objective[batch_idx, 7, row, col] = math.sin(direction1)/2 + 0.5
            # Marking Point Type Regression
            objective[batch_idx, 8, row, col] = marking_point.type
            # Assign Gradient
            gradient[batch_idx, 1:9, row, col].fill_(2.)
            gradient[batch_idx, 4:8, row, col].fill_(6.)
    return objective, gradient

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_detector(args):
    """Train directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str('1') if args.cuda else 'cpu')
    torch.set_grad_enabled(True)

    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL)
    
    
    dp_detector.to(device)
    if args.detector_weights:
        print("Loading weights: %s" % args.detector_weights)
        dp_detector.load_state_dict(torch.load(args.detector_weights))
    dp_detector.train()

    optimizer = torch.optim.Adam(dp_detector.parameters(), 1e-4)
    # optimizer = torch.optim.SGD(dp_detector.parameters(), 1e-4)

    if args.optimizer_weights:
        print("Loading weights: %s" % args.optimizer_weights)
        optimizer.load_state_dict(torch.load(args.optimizer_weights))

    logger = util.Logger(args.enable_visdom, ['train_loss'])

    file_path = 'yaml/data_root.yaml'

    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    data_root = yaml_data['data_root']
    point_train = yaml_data['point_train']
    path = data_root + point_train

    data_loader = DataLoader(data.ParkingSlotDataset(path),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)))

    for epoch_idx in range(args.num_epochs):
        for iter_idx, (images, marking_points) in enumerate(data_loader):
            images = torch.stack(images).to(device)
            optimizer.zero_grad()
            prediction = dp_detector(images)
            objective, gradient = generate_objective(marking_points, device, True)
            loss = (prediction - objective) ** 2
            loss.backward(gradient)
            optimizer.step()

            train_loss = torch.sum(loss*gradient).item() / loss.size(0)
            logger.log(epoch=epoch_idx, iter=iter_idx, train_loss=train_loss)
            if args.enable_visdom:
                plot_prediction(logger, images, marking_points, prediction)
        
        torch.save(dp_detector.state_dict(),
                   'weights/augbig/dmpr_%d.pth' % (epoch_idx))
        
        # torch.save(optimizer.state_dict(), 'weights/optimizer.pth')

import traceback
if __name__ == '__main__':
    same_seeds(114514)
    try:
        train_detector(config.get_parser_for_training().parse_args())
    except Exception as e:
        fp = open('log.txt', 'a')
        traceback.print_exc(file=fp)
        fp.close()
