#!/usr/bin/env python3
"""Process an image with the trained neural network
Usage:
    demo.py [options] <yaml-config> <checkpoint> <image> <output-svg>
    demo.py (-h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <image>                       Path to the input image
   <output-svg>                  Path to save the SVG output

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
"""

import os
import os.path as osp
import pprint
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import yaml
from docopt import docopt
from sklearn.cluster import DBSCAN

import glsp
from glsp.config import C, M
from glsp.models.line_vectorizer import LineVectorizer
from glsp.models.multitask_learner import MultitaskHead, MultitaskLearner
from glsp.postprocess import postprocess
from glsp.utils import recursive_to

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    # Add node clustering parameters
    NODE_CLUSTER_EPS = 20  # Increased from 3 to 5 pixels to better handle close nodes
    NODE_CLUSTER_MIN_SAMPLES = 1  # Keep this at 1 to ensure all points are considered

    # NEW: Get output path if provided
    output_svg = args["<output-svg>"]

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)
    checkpoint = torch.load(args["<checkpoint>"], map_location=device)

    # Load model
    model = glsp.models.hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
    )
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    imname = args["<image>"]
    print(f"Processing {imname}")
    im = skimage.io.imread(imname)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, 2)
    im = im[:, :, :3]
    im_resized = skimage.transform.resize(im, (512, 512)) * 255
    image = (im_resized - M.image.mean) / M.image.stddev
    image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
    with torch.no_grad():
        input_dict = {
            "image": image.to(device),
            "meta": [
                {
                    "junc": torch.zeros(1, 2).to(device),
                    "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                    "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                    "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                }
            ],
            "target": {
                "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
            },
            "mode": "testing",
        }
        H = model(input_dict)["preds"]

    lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
    scores = H["score"][0].cpu().numpy()
    for i in range(1, len(lines)):
        if (lines[i] == lines[0]).all():
            lines = lines[:i]
            scores = scores[:i]
            break

    diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
    nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

    # Save only one SVG at the specified path, with threshold 0.94
    t = 0.94
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    # Collect all unique node positions
    all_nodes = []
    for (a, b), s in zip(nlines, nscores):
        if s < t:
            continue
        all_nodes.extend([a, b])
    
    # Convert to numpy array for clustering
    all_nodes = np.array(all_nodes)
    
    # Cluster nodes using DBSCAN
    clustering = DBSCAN(eps=NODE_CLUSTER_EPS, min_samples=NODE_CLUSTER_MIN_SAMPLES).fit(all_nodes)
    
    # Get unique cluster centers with improved handling
    cluster_centers = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:  # Skip noise points
            continue
        cluster_points = all_nodes[clustering.labels_ == cluster_id]
        # Use weighted average based on distance to ensure better centering
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)
    
    cluster_centers = np.array(cluster_centers)

    # Plot lines and nodes using the clustered centers
    for (a, b), s in zip(nlines, nscores):
        if s < t:
            continue
        # Find closest cluster centers for each endpoint with improved matching
        a_center = cluster_centers[np.argmin(np.linalg.norm(cluster_centers - a, axis=1))]
        b_center = cluster_centers[np.argmin(np.linalg.norm(cluster_centers - b, axis=1))]
        
        # Only plot if the centers are not too close to each other
        if np.linalg.norm(a_center - b_center) > NODE_CLUSTER_EPS:
            plt.plot([a_center[1], b_center[1]], [a_center[0], b_center[0]], c=c(s), linewidth=2, zorder=s)
            plt.scatter(a_center[1], a_center[0], **PLTOPTS)
            plt.scatter(b_center[1], b_center[0], **PLTOPTS)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(im)
    plt.savefig(output_svg, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
