# GLSP: GNN-based Line Segment Parser

GLSP (Global Line Segment Proposal) is a **GNN-based** system for extracting vectorized wireframes (lines and junctions) from images such as architectural drawings, floor plans, or technical diagrams.

## What does it do?
- Detects junctions and line segments in images using graph neural network techniques
- Outputs a clean, vectorized wireframe (SVG)
- Useful for digitizing sketches, blueprints, and diagrams

## How to Run

### 1. **Docker (Recommended)**

```
docker-compose up --build
```
- This will start the API server on port 8000 (see `docker-compose.yml`).

### 2. **Manual (Python)**

Install dependencies:
```
pip install -r requirements.txt
```

Run the API server:
```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. **Process an Image (Command Line)**

```
python run.py config/wireframe.yaml checkpoints/model.pth <input_image> <output_svg>
```

---

- For API usage, POST an image to `/process/` and receive an SVG wireframe in response.
- See `testing.py` for a simple example of how to call the API from Python. 