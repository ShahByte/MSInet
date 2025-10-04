import time
import numpy as np
import torch
from torch.autograd import Variable
from model import MyNet
from utilis import superpixels, create_superpixel_indices, norm_array
from utilis import reduce_dimensions_umap
from utilis import superpixel_refinement_1
from config import args
from training import train_model
import argparse

use_cuda = torch.cuda.is_available()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run MSInet training')
    parser.add_argument('--data', type=str, required=True, help='Path to input dataset file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')

    args = parser.parse_args()

    # Load data
    load_image = np.load(args.data)
    data_name = 'simulated_'
    load_image = norm_array(load_image)
    im = load_image.reshape(70, 70, 3)

    data = torch.from_numpy(
        np.array([im.transpose(2, 0, 1).astype('float32') / 255.])
    )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    # Superpixels
    in_image_fr_sp = load_image
    sp_map = superpixels(in_image_fr_sp, im.shape[0], im.shape[1], im.shape[2])
    sp_indices_map = create_superpixel_indices(sp_map)
    labels = [np.array(indices) for indices in sp_indices_map.values()]

    # Model
    model = MyNet(data.size(1))
    if use_cuda:
        model.cuda()
    model.train()

    # Start training
    train_model(model, data, im, labels, data_name)


if __name__ == "__main__":
    main()
