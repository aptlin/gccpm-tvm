import argparse
import os
import time

import cv2
import numpy as np
import torch
import tvm
from tvm import relay
from tvm.contrib import util

from modules import load_single_pose_model
from run_utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["CocoSingle", "Lip"],
    default="CocoSingle",
    help="name dataset for validation: <Lip> or <CocoSingle>",
)
parser.add_argument("--output-dir", type=str, default="./scratchpad/build")
parser.add_argument("--input-name", type=str, default="image")
parser.add_argument("--force-cpu", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--processing-width", type=int, default=600)
parser.add_argument("--processing-height", type=int, default=340)
parser.add_argument("--n-channels", type=int, default=3)
parser.add_argument("--target", type=str, default="llvm")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.force_cpu
    else torch.device("cpu")
)


def main():
    model = load_single_pose_model(args.model)
    model = model.to(DEVICE)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    start = time.time()
    random_cv2_image = np.random.randint(
        256,
        size=(1, args.n_channels, args.processing_height, args.processing_width),
        dtype=np.uint8,
    )

    # TODO: try resizing to a smaller image
    input_image = random_cv2_image

    scripted_model = torch.jit.trace(model, torch.Tensor(input_image)).eval()

    input_name = args.input_name
    shape_list = [(input_name, input_image.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    target = tvm.target.create(args.target)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, target=target, params=params)

    path_lib = "{}/deploy_lib_{}_{}_{}.tar".format(
        args.output_dir, args.input_name, args.processing_width, args.processing_height
    )
    path_graph = "{}/deploy_graph_{}_{}_{}.json".format(
        args.output_dir, args.input_name, args.processing_width, args.processing_height
    )
    path_params = "{}/deploy_params_{}_{}_{}.params".format(
        args.output_dir, args.input_name, args.processing_width, args.processing_height
    )

    lib.export_library(path_lib)
    with open(path_graph, "w") as f:
        f.write(graph)
    with open(path_params, "wb") as f:
        f.write(relay.save_param_dict(params))

    ctx = tvm.cpu(0) if str(DEVICE) == "cpu" else tvm.gpu()
    print("-" * 80)
    print("Done! Converted and serialized the model to:")
    print("\t- lib: {}".format(path_lib))
    print("\t- graph: {}".format(path_graph))
    print("\t- params: {}".format(path_params))
    print("-" * 80)


if __name__ == "__main__":
    main()
