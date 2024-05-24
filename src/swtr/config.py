from yacs.config import CfgNode as CN

sushi = CN()

sushi.MODEL = CN()
sushi.MODEL.SWIN = CN()
sushi.MODEL.SWIN.PATCH_SIZE = 4
sushi.MODEL.SWIN.IN_CHANS = 3
sushi.MODEL.SWIN.EMBED_DIM = 96
sushi.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
sushi.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
sushi.MODEL.SWIN.WINDOW_SIZE = 7
sushi.MODEL.SWIN.MLP_RATIO = 4.
sushi.MODEL.SWIN.QKV_BIAS = True
sushi.MODEL.SWIN.QK_SCALE = None
sushi.MODEL.SWIN.APE = False
sushi.MODEL.SWIN.PATCH_NORM = True

sushi.MODEL.SWIN_MLP = CN()
sushi.MODEL.SWIN_MLP.PATCH_SIZE = 4
sushi.MODEL.SWIN_MLP.IN_CHANS = 3
sushi.MODEL.SWIN_MLP.EMBED_DIM = 96
sushi.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
sushi.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
sushi.MODEL.SWIN_MLP.WINDOW_SIZE = 7
sushi.MODEL.SWIN_MLP.MLP_RATIO = 4.
sushi.MODEL.SWIN_MLP.APE = False
sushi.MODEL.SWIN_MLP.PATCH_NORM = True

def get_config():
    return sushi.clone()
