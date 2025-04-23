from flask import Flask, request, jsonify
import os
import torch
import skimage.io
import skimage.transform
import numpy as np
from glsp.config import C, M
from glsp.models.line_vectorizer import LineVectorizer
from glsp.models.multitask_learner import MultitaskHead, MultitaskLearner
from glsp.postprocess import postprocess
import glsp.models
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global variables for model and device
model = None
device = None

def init_model():
    global model, device
    
    # Load configuration
    config_path = os.getenv('CONFIG_PATH', 'config/wireframe.yaml')
    C.update(C.from_yaml(filename=config_path))
    M.update(C.model)

    # Set up device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

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
    
    # Load checkpoint
    checkpoint_path = os.getenv('MODEL_PATH', 'checkpoints/model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

def process_image(image_path):
    # Read and preprocess image
    im = skimage.io.imread(image_path)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, 2)
    im = im[:, :, :3]
    im_resized = skimage.transform.resize(im, (512, 512)) * 255
    image = (im_resized - M.image.mean) / M.image.stddev
    image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()

    # Process with model
    with torch.no_grad():
        input_dict = {
            "image": image.to(device),
            "meta": [{
                "junc": torch.zeros(1, 2).to(device),
                "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
            }],
            "target": {
                "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
            },
            "mode": "testing",
        }
        H = model(input_dict)["preds"]

    # Process results
    lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
    scores = H["score"][0].cpu().numpy()
    
    # Clean up duplicates
    for i in range(1, len(lines)):
        if (lines[i] == lines[0]).all():
            lines = lines[:i]
            scores = scores[:i]
            break

    # Post-process
    diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
    nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)
    
    return nlines, nscores, im

def plot_results(im, nlines, nscores, threshold=0.97):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    
    # Plot lines above threshold
    for (a, b), s in zip(nlines, nscores):
        if s < threshold:
            continue
        plt.plot([a[1], b[1]], [a[0], b[0]], 'r-', linewidth=2)
        plt.scatter([a[1], b[1]], [a[0], b[0]], color='cyan', s=15)
    
    plt.axis('off')
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    threshold = float(request.form.get('threshold', 0.97))
    
    # Save uploaded file
    temp_path = os.path.join(os.getenv('INPUT_DIR', 'images'), file.filename)
    file.save(temp_path)
    
    try:
        # Process image
        nlines, nscores, im = process_image(temp_path)
        
        # Generate visualization
        buf = plot_results(im, nlines, nscores, threshold)
        
        # Convert to base64
        img_str = base64.b64encode(buf.getvalue()).decode()
        
        # Save results
        output_path = os.path.join(os.getenv('OUTPUT_DIR', 'output'), f'result_{file.filename}')
        with open(output_path, 'wb') as f:
            f.write(buf.getvalue())
        
        return jsonify({
            'success': True,
            'image': img_str,
            'lines': nlines.tolist(),
            'scores': nscores.tolist(),
            'output_path': output_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    init_model()
    app.run(host='0.0.0.0', port=8000) 