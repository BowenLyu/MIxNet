import argparse
import os
import GPUtil


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default='config/config.yaml', help='training configuration')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU index')

    args = parser.parse_args()

    # assign GPU
    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = args.gpu
    GPU_INDEX = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(GPU_INDEX)

    from training.trainer import Basetrainner

    trainrunner = Basetrainner(args)
    trainrunner.run() 