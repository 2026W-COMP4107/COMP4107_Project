import gradio as gr
from PIL import Image
import requests
from io import BytesIO
from utils import classify_image, generate_gradcam


def process(image, url):
    if image is None and (not url or not url.strip()):
        raise gr.Error("Provide an image or a URL.")

    if image is None:
        try:
            resp = requests.get(url.strip(), timeout=10)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content))
        except Exception as e:
            raise gr.Error(f"Failed to load image from URL: {e}")

    class_name, confidence, probs = classify_image(image)
    gradcam_img = generate_gradcam(image)

    display_img = image.convert("RGB").resize((224, 224))

    result = (
        f"Prediction: {class_name}\n"
        f"Confidence: {confidence:.2%}\n\n"
        f"clean:        {probs['clean']:.2%}\n"
        f"spaghetti: {probs['spaghetti']:.2%}"
    )
    return display_img, gradcam_img, result


with gr.Blocks(title="Spaghetti Detector") as app:
    gr.Markdown("# Spaghetti Detector")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil", label="Upload or Drag & Drop an image"
            )
            url_input = gr.Textbox(
                label="Or paste a URL",
                placeholder="link to image",
            )
            btn = gr.Button("Check", variant="primary", size="lg")

        with gr.Column(scale=1):
            with gr.Row():
                result_image = gr.Image(label="Input")
                gradcam_image = gr.Image(label="Grad-CAM")
            result_text = gr.Textbox(label="Results", lines=6, interactive=False)

    btn.click(
        fn=process,
        inputs=[image_input, url_input],
        outputs=[result_image, gradcam_image, result_text],
    )

app.launch(theme=gr.themes.Soft(), server_name="0.0.0.0")
