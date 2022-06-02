import sys

sys.path.append("detr")
sys.path.append("src")

from typing import Tuple

import gradio as gr
import numpy as np
import pandas as pd
from models.custom import TableRecogniser
from PIL import Image

"""
Gradio app to demo Table Extraction task using Table-Transformer from Microsoft.

(env)$ python3 gradio_app.py

# head to http://sota-ip:7860/

"""


examples = [
    "gradio_examples/BHP-04.jpg",
    "gradio_examples/BHP-08.jpg",
    "gradio_examples/BHP-11.jpg",
]

description = (
    "- Official Microsoft Table-Transformer Github Page: https://github.com/microsoft/table-transformer \n"
    "- PubTables-1M: Towards comprehensive table extraction from unstructured documents: https://arxiv.org/abs/2110.00061 \n"
    "- GriTS: Grid table similarity metric for table structure recognition: https://arxiv.org/abs/2203.12555"
)

model = TableRecogniser()


def predict(inp: Image.Image) -> Tuple[pd.DataFrame, np.ndarray]:
    outputs = model.predict(inp)
    dfs, imgs = outputs
    # doesn't support dynamic size of dataframe output?
    return dfs[0], imgs[0]


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Input"),
    outputs=[
        gr.Dataframe(type="pandas", label="Dataframe"),
        gr.Image(type="pil", label="Visualisation"),
    ],
    title="[Table Recognition] Image Page to Dataframe",
    description=description,
    examples=examples,
).launch(
    debug=True, share=False, server_name="0.0.0.0", server_port=7860, show_error=True
)
