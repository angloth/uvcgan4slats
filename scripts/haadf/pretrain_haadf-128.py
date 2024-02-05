import argparse
import os

from uvcgan import ROOT_DATA, ROOT_OUTDIR, train
from uvcgan.utils.parsers import add_preset_name_parser, add_batch_size_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Pretrain HAADF-128 BERT')
    add_preset_name_parser(parser, 'gen', GEN_PRESETS, 'uvcgan_128', help_msg='generator type')
    add_batch_size_parser(parser, default = 64)
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

if __name__ == "__main__":

    cmdargs   = parse_cmdargs()
    args_dict = {
        'batch_size' : cmdargs.batch_size,
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
        'epochs'        : 499,
        'discriminator' : None,
        'generator' : {
            **GEN_PRESETS[cmdargs.gen],
            'optimizer'  : {
                'name'  : 'AdamW',
                'lr'    : cmdargs.batch_size * 5e-5 / 512,
                'betas' : (0.9, 0.99),
                'weight_decay' : 0.05,
            },
            'weight_init' : {
                'name'      : 'normal',
                'init_gain' : 0.02,
            }
        },
        'model'      : 'autoencoder',
        'model_args' : {
            'joint'   : True,
            'background_penalty' : {
                'epochs_warmup' : 25,
                'epochs_anneal' : 75,
            },
            'masking' : {
                'name'       : 'image-patch-random',
                'patch_size' : (16, 16), # Perhaps these are too small? Original is (32, 32) but is halved due to the input size being half
                'fraction'   : 0.4,
            },
        },
        'scheduler' : {
            'name'    : 'CosineAnnealingWarmRestarts',
            'T_0'     : 100,
            'T_mult'  : 1,
            'eta_min' : cmdargs.batch_size * 5e-5 * 1e-5 / 512,
        },
        'loss'             : 'l2',
        'gradient_penalty' : None,
        'steps_per_epoch'  : 32 * 1024 // cmdargs.batch_size,
    # args
        'label'      : 'pretrain-haadf-128',
        'outdir'     : os.path.join(ROOT_OUTDIR, 'haadf'),
        'log_level'  : 'DEBUG',
        'checkpoint' : 100,
    }

    train(args_dict)
