import argparse
import os

from uvcgan import ROOT_OUTDIR, train
from uvcgan.utils.parsers import add_preset_name_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Train HAADF translation with uvcgan'
    )

    add_preset_name_parser(parser, 'gen',  GEN_PRESETS, 'uvcgan_128', help_msg='generator type')

    parser.add_argument('--labmda-cycle',
                        dest = 'lambda_cyc',
                        type = float,
                        default = 1.0,
                        help = 'magnitude of the cycle-consisntecy loss (default = 1.0)')

    parser.add_argument('--lr-disc',
                        dest = 'lr_disc',
                        type = float,
                        default = 5e-5,
                        help = 'learning rate of the discriminator (default = 5e-5)')

    parser.add_argument('--lr-gen',
                        dest = 'lr_gen',
                        type = float,
                        default = 1e-5,
                        help = 'learning rate of the generator (default = 1e-5)')

    parser.add_argument('--gp-constant',
                        dest = 'constant_gp',
                        type = float,
                        default = 10.,
                        help = ('the constant gamma for gradient penalty (default = 10). '
                                'See the UVCGAN paper (https://arxiv.org/pdf/2203.02557.pdf) '
                                'section 3.3 for more detail.'))

    parser.add_argument('--gp-lambda',
                        dest = 'lambda_gp',
                        type = float,
                        default = 1.,
                        help = ('the coefficient of the gradient penalty (default = 1). '
                                'See the UVCGAN paper (https://arxiv.org/pdf/2203.02557.pdf) '
                                'section 3.3 for more detail.'))

    return parser.parse_args()

GEN_PRESETS = {
    'resnet9' : {
        'model'      : 'resnet_9blocks',
        'model_args' : None,
    },
    'unet' : {
        'model'      : 'unet_256',
        'model_args' : None,
    },
    'resnet9-nonorm' : {
        'model'      : 'resnet_9blocks',
        'model_args' : {
            'norm' : 'none',
        },
    },
    'unet-nonorm' : {
        'model'      : 'unet_256',
        'model_args' : {
            'norm' : 'none',
        },
    },
    'uvcgan' : {
        'model' : 'vit-unet',
        'model_args' : {
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 12,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            'unet_features_list' : [58, 96, 192, 384],
            'unet_activ'         : 'leakyrelu',
            'unet_norm'          : None,
            'unet_downsample'    : 'conv',
            'unet_upsample'      : 'upsample-conv',
            'rezero'             : True,
            'activ_output'       : None,
        },
    },
    'uvcgan_128' : {
        'model' : 'vit-unet',
        'model_args' : {
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 12,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            'unet_features_list' : [96, 192, 384], # If the model overfits, try [58, 96, 192] and change embed_features and features to 192
            'unet_activ'         : 'leakyrelu',
            'unet_norm'          : None,
            'unet_downsample'    : 'conv',
            'unet_upsample'      : 'upsample-conv',
            'rezero'             : True,
            'activ_output'       : None,
        },
    },
}

cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : 1,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'ndarray-domain-hierarchy',
                    'domain' : domain,
                    'path'   : 1/0 #TODO: Set path to HAADF dataset @benke :),
                },
                'shape'           : (1, 128, 128),
                'transform_train' : None,
                'transform_test'  : None,
            } for domain in [ 'fake', 'real' ]
        ],
        'merge_type' : 'unpaired',
    },
    'epochs'        : 500,
    'discriminator' : {
        'model' : 'basic',
        'model_args' : None,
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr_disc,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr_gen,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'cyclegan',
    'model_args' : {
        'lambda_a'   : cmdargs.lambda_cyc,
        'lambda_b'   : cmdargs.lambda_cyc,
        'lambda_idt' : 0.5,
        'pool_size'  : 50,
    },
    'scheduler' : {
        'name'          : 'linear',
        'epochs_warmup' : 250,
        'epochs_anneal' : 250,
    },
    'loss' : 'lsgan',
    'gradient_penalty' : {
        'constant'  : cmdargs.constant_gp,
        'lambda_gp' : cmdargs.lambda_gp / (cmdargs.constant_gp ** 2),
    },
    'steps_per_epoch'  : 2000,
    'transfer' : {
        'base_model'   : (
            'haadf/model_m(autoencoder)_d(None)_g(vit-unet)_pretrain-haadf-128'
        ),
        'transfer_map' : {
            'gen_ab' : 'encoder',
            'gen_ba' : 'encoder',
        },
        'strict'        : True,
        'allow_partial' : False,
    },
# args
    'label' : (
        f'train-{cmdargs.gen}'
        f'-({cmdargs.lambda_cyc}:{cmdargs.lr_gen}:{cmdargs.lr_disc})'
        '_haadf-128'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'haadf'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 50,
}

train(args_dict)

