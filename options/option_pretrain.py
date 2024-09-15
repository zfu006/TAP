import argparse

class BaseOptions():
    def initialize(self):
        """Initialize options"""
        parser = argparse.ArgumentParser()

        # basic settings
        parser.add_argument('--save_logs_dir', type=str, default='./saves/logs/', help='specify the save dir for logs, common path is "./saves/exp/" ')
        parser.add_argument('--save_imgs_dir', type=str, default='./saves/imgs/', help='specify the save dir for imgs, common path is "./saves/exp/"')
        parser.add_argument('--save_models_dir', type=str, default='./saves/models/', help='specify the save dir for models, common path is "./saves/exp/"')
        parser.add_argument('--train_clean_img_dir', type=str, default='./Datasets/pretrain_cropped_images/raw_rgb/', help='clean image path for paired training data')
        parser.add_argument('--test_clean_img_dir', type=str, default='./Datasets/CRVD_dataset/indoor_raw_gt/', help='val img root')
        parser.add_argument('--test_noisy_img_dir', type=str, default='./Datasets/CRVD_dataset/indoor_raw_noisy/')
        parser.add_argument('--noise_aug', action='store_true', help='set whether to augment noise model')
        parser.add_argument('--bayer_aug', action='store_true', help='set whether to augment bayer pattern')
        parser.add_argument('--datasets', type=str, nargs='*', help='select training datasets, dataset names: [SID, SIDD, CRVD, Sensenoise]')
        parser.add_argument('--patch_size', type=int, default=128, help='cropped patch size')        
        parser.add_argument('--train_batch_size', type=int, default=8, help='set batch size for training')
        parser.add_argument('--test_batch_size', type=int, default=4, help='set batch size for testing')
        parser.add_argument('--not_train_shuffle', action='store_true', help='set shuffle scheme for training, if set the flag, the dataset will not be shuffled')
        parser.add_argument('--not_train_drop_last', action='store_true', help='set drop last scheme for training, if set the flag, the dataset will drop the last files')
        parser.add_argument('--num_workers', type=int, default=4, help='set number of workers to load data')
        parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='set gpu for training, e.g. 0 or 1 for single gpu training and 0 1 for multiple gpus training')

        # network and training settings
        parser.add_argument('--in_nc', type=int, default=4, help='input channels of the network')  
        parser.add_argument('--out_nc', type=int, default=4, help='output channels of the network') 
        parser.add_argument('--nc', type=int, default=64, help='set the base number of channels for the network')
        parser.add_argument('--verbose', action='store_true', help='set whether to print the network architecture for debug')        
        parser.add_argument('--init_type', type=str, default='normal', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='set the network initiate method')
        parser.add_argument('--G_lr', type=float, default=1e-3, help='set the optimizer_G learning rate')
        parser.add_argument('--pixel_loss_type', type=str, default='L1', help='set pixel level loss type')  
        parser.add_argument('--epochs', type=int, default=500, help='set training epochs')
        parser.add_argument('--test_step', type=int, default=500, help='set testing steps')
        parser.add_argument('--save_models', action='store_true', help='set whether to save models')
        
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print options in the log"""
        message = ''
        message += '--------------- Options ---------------\n'

        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: {}]'.format(str(default))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '--------------- End ---------------\n'
        print(message)

        return message
        
    
    def parse(self):
        opt = self.initialize() 
        opt_message = self.print_options(opt) 
        
        return opt, opt_message

if __name__ == '__main__':
    BaseOptions().parse()