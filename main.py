import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import wandb

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    mydata_loader = get_loader(
        config.mydata_image_dir,
        config.mydata_attr_path,
        config.selected_attrs,
        config.mydata_crop_size,
        config.image_size,
        config.batch_size,
        config.mode,
        config.num_workers
    )

    if config.mode != 'test':
        test_loader = get_loader(
            config.test_dataset,
            config.mydata_attr_path,
            config.selected_attrs,
            config.mydata_crop_size,
            config.image_size,
            batch_size=1,
            mode='test',
            num_workers=config.num_workers
        )
    else:
        test_loader = None

    solver = Solver(mydata_loader, test_loader, config)

    if config.mode == 'train':
        wandb.init(
            project="StarGAN-R2R",
            name="R2R-4-3",
            config={
                "g_lr":       solver.g_lr,
                "d_lr":       solver.d_lr,
                "batch_size": solver.batch_size,
                "num_iters":  solver.num_iters,
                "lambda_cls": solver.lambda_cls,
                "lambda_rec": solver.lambda_rec,
                "lambda_gp":  solver.lambda_gp,
                "lambda_id": solver.lambda_id,
                "n_critic":   solver.n_critic,
            }
        )
        solver.train()
    else:
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test StarGAN on MyData')

    # Model configuration
    parser.add_argument('--c_dim', type=int, default=2,
                        help='dimension of domain labels for MyData')
    parser.add_argument('--mydata_crop_size', type=int, default=256,
                        help='crop size for MyData images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='input image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64,
                        help='number of conv filters in first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64,
                        help='number of conv filters in first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6,
                        help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6,
                        help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1,
                        help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10,
                        help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10,
                        help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=5.0,
                    help='weight for identity loss')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=400000,
                        help='total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000,
                        help='iterations to start decaying learning rate')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='number of D updates per G update')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    parser.add_argument('--resume_iters', type=int, default=None,
                        help='iteration to resume training from')
    parser.add_argument('--selected_attrs', nargs='+',
                        default=['cameraA', 'cameraB'],
                        help='list of attributes for MyData')

    # Test configuration
    parser.add_argument('--test_iters', type=int, default=200000,
                        help='iteration to load model for testing')

    # Miscellaneous
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of data loader workers')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='mode: train or test')
    parser.add_argument('--use_tensorboard', type=str2bool, default=True,
                        help='whether to use TensorBoard')

    # Directories
    parser.add_argument('--mydata_image_dir', type=str,
                        default='data/mydata/images',
                        help='directory for MyData images')
    parser.add_argument('--mydata_attr_path', type=str,
                        default='data/mydata/list_attr_mydata.txt',
                        help='path to MyData attribute file')
    parser.add_argument('--test_dataset', type=str, default='data/mydata/list_attr_mydata.txt', help='test')
    parser.add_argument('--log_dir', type=str, default='mydata/logs',
                        help='directory to save training logs')
    parser.add_argument('--model_save_dir', type=str, default='mydata/models',
                        help='directory to save model checkpoints')
    parser.add_argument('--sample_dir', type=str, default='mydata/samples',
                        help='directory to save sample images during training')
    parser.add_argument('--result_dir', type=str, default='mydata/results',
                        help='directory to save test results')

    # Step intervals
    parser.add_argument('--log_step', type=int, default=10,
                        help='interval for logging')
    parser.add_argument('--sample_step', type=int, default=1000,
                        help='interval for saving sample images')
    parser.add_argument('--model_save_step', type=int, default=10000,
                        help='interval for saving model checkpoints')
    parser.add_argument('--lr_update_step', type=int, default=1000,
                        help='interval for updating learning rate')
    parser.add_argument('--eval_step', type=int, default=5000,
                        help='interval for evaluating model')

    config = parser.parse_args()
    print(config)
    main(config)