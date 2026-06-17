import gradio as gr
import src.noise_models as ns
import src.morphological as morph
import src.smoothing_filters as sm
import src.edge_detection as ed
import src.segmentation as seg
import src.colorspace as cs
import src.enhancement as enh
import utils.helpers as hp

CATEGORY_MAPPING = {
    "Morphological Operations": [
        "Erosion", "Dilation", "Opening", "Closing",
        "Gradient", "Top Hat", "Black Hat"
    ],
    "Edge Detection Filters": [
        "Canny Edge Detector", "Sobel Filter", "Prewitt Edge Detector",
        "Roberts Cross Detector", "Laplacian of Gaussian (LoG)"
    ],
    "Segmentation & Clustered Bins": [
        "Multi-Otsu", "Global Thresholding", "Otsu's Method",
        "Adaptive Gaussian", "Adaptive Mean", "K-Means Clustering"
    ],
    "Intensity Transforms & Enhancements": [
        "Gamma Correction", "Negative / Inversion", "Log Transform", "Histogram Equalization", "CLAHE", "Contrast Stretching"
    ],
    "Smoothing Filters (Spatial Domain)": [
        "Gaussian Filter", "Mean Filter (Blur)", "Median Filter", "Bilateral Filter"
    ],
    "Noise Models": [
        "Salt & Pepper Noise", "Gaussian Noise", "Speckle Noise", "Periodic Noise"
    ],
    "Color Space & Transforms": [
        "RGB to Grayscale", "Simulate Deuteranopia", "RGB to HSV", "White Balance"
    ]
}


def core_routing_pipeline(
        input_img, category, sub_op, kernel_size, param1, param2, transform_param, noise_param, color_w_red
):
    if input_img is None:
        return None, None

    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # ── ROUTING LOGIC ──
    if category == "Morphological Operations":
        if sub_op == "Erosion":
            out = morph.erosion(input_img, kernel_size)
        elif sub_op == "Dilation":
            out = morph.dilation(input_img, kernel_size)
        elif sub_op == "Opening":
            out = morph.opening(input_img, kernel_size)
        elif sub_op == "Closing":
            out = morph.closing(input_img, kernel_size)
        elif sub_op == "Gradient":
            out = morph.morphological_gradient(input_img, kernel_size)
        elif sub_op == "Top Hat":
            out = morph.top_hat(input_img, kernel_size)
        elif sub_op == "Black Hat":
            out = morph.black_hat(input_img, kernel_size)
        else:
            out = input_img

    elif category == "Edge Detection Filters":
        if sub_op == "Sobel Filter":
            out = ed.sobel_edges(input_img, ksize=3)
        elif sub_op == "Prewitt Edge Detector":
            out = ed.prewitt_edges(input_img)
        elif sub_op == "Roberts Cross Detector":
            out = ed.roberts_edges(input_img)
        elif sub_op == "Canny Edge Detector":
            out = ed.canny_edges(input_img, low=int(param1), high=int(param2))
        elif sub_op == "Laplacian of Gaussian (LoG)":
            out = ed.log_edges(input_img, sigma=1.0, threshold=int(param1))
        else:
            out = input_img

    elif category == "Segmentation & Clustered Bins":
        if sub_op == "Multi-Otsu":
            out = seg.multi_otsu(input_img, target_classes=int(param1))
        elif sub_op == "Global Thresholding":
            out = seg.global_threshold(input_img, thresh=int(param1))
        elif sub_op == "Otsu's Method":
            out = seg.otsu_threshold(input_img)
        elif sub_op == "Adaptive Gaussian":
            out = seg.adaptive_threshold(input_img, method_type="gaussian", block_size=int(param1))
        elif sub_op == "Adaptive Mean":
            out = seg.adaptive_threshold(input_img, method_type="mean", block_size=int(param1))
        elif sub_op == "K-Means Clustering":
            out = seg.kmeans_segmentation(input_img, k=int(param1))
        else:
            out = input_img


    elif category == "Intensity Transforms & Enhancements":

        if sub_op == "CLAHE":
            out = enh.apply_clahe(input_img, clip_limit=transform_param, tile_grid_size=param1)
        elif sub_op == "Contrast Stretching":
            out = enh.contrast_stretching(input_img, r1=int(param1), s1=0, r2=int(param2), s2=255)
        elif sub_op == "Negative / Inversion":
            out = enh.inversion(input_img)
        elif sub_op == "Log Transform":
            out = enh.log_transform(input_img, transform_param)
        elif sub_op == "Gamma Correction":
            out = enh.gamma_correction(input_img, gamma=transform_param)
        elif sub_op == "Histogram Equalization":
            out = enh.histogram_equalization(input_img)
        else:
            out = input_img

    elif category == "Smoothing Filters (Spatial Domain)":
        if sub_op == "Mean Filter (Blur)":
            out = sm.mean_filter(input_img, kernel_size)
        elif sub_op == "Gaussian Filter":
            out = sm.gaussian_filter(input_img, kernel_size, sigma=0)
        elif sub_op == "Median Filter":
            out = sm.median_filter(input_img, kernel_size)
        elif sub_op == "Bilateral Filter":
            out = sm.bilateral_filter(input_img, kernel_size)
        else:
            out = input_img

    elif category == "Noise Models":
        if sub_op == "Gaussian Noise":
            out = ns.add_gaussian_noise(input_img, sigma=noise_param)
        elif sub_op == "Salt & Pepper Noise":
            out = ns.add_salt_and_pepper(input_img, density=(noise_param / 100.0))
        elif sub_op == "Speckle Noise":
            out = ns.add_speckle_noise(input_img, variance=(noise_param / 100.0))
        elif sub_op == "Periodic Noise":
            out = ns.add_periodic_noise(input_img, frequency=int(noise_param))
        else:
            out = input_img

    elif category == "Color Space & Transforms":
        if sub_op == "RGB to Grayscale":
            out = cs.rgb_to_grayscale(input_img)
        elif sub_op == "Simulate Deuteranopia":
            out = cs.simulate_deuteranopia(input_img)
        elif sub_op == "RGB to HSV":
            out = cs.rgb_to_hsv_view(input_img)
        elif sub_op == "White Balance":
            out = cs.white_balance(input_img)
        else:
            out = input_img
    else:
        out = input_img

    hist_data = hp.generate_live_histogram(out)
    return out, hist_data

#  GRADIO INTERFACE LAYOUT
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Modular Image Processing Web Station")

    with gr.Row():
        with gr.Column(scale=1):
            source_image = gr.Image(type="numpy", label="Upload Source Image")

            category_dropdown = gr.Dropdown(
                choices=list(CATEGORY_MAPPING.keys()),
                value="Segmentation & Clustered Bins",
                label="Choose Pipeline Category Framework"
            )

            sub_op_dropdown = gr.Dropdown(
                choices=CATEGORY_MAPPING["Segmentation & Clustered Bins"],
                value="Multi-Otsu",
                label="Sub-Operation Method Select"
            )

            kernel_slider = gr.Slider(minimum=3, maximum=21, step=2, value=5, label="Kernel Structural Size Window",
                                      visible=False)
            param1_slider = gr.Slider(minimum=3, maximum=5, step=1, value=3, label="Target Region Classes (Bins)",
                                      visible=True)
            param2_slider = gr.Slider(minimum=0, maximum=255, step=1, value=150, label="High Hysteresis Threshold",
                                      visible=False)
            transform_slider = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, value=1.0,
                                         label="Transform Factor Scale Multiplier", visible=False)
            noise_slider = gr.Slider(minimum=1, maximum=100, step=1, value=5, label="Noise Density Factor/Sigma Value",
                                     visible=False)
            color_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.299,
                                     label="Custom Grayscale Red-Channel Weight Ratio", visible=False)

            submit_btn = gr.Button("Deploy Pipeline Execution Engine", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="numpy", label="Processed Output Target View")
            histogram_output = gr.Image(type="pil", label="Output Dynamic Matrix Histogram")


    def on_category_change(cat):
        sub_ops = CATEGORY_MAPPING[cat]
        default_val = sub_ops[0]

        show_kernel = cat in ["Morphological Operations", "Smoothing Filters (Spatial Domain)"]
        show_param1 = cat in ["Edge Detection Filters", "Segmentation & Clustered Bins"]
        show_param2 = (cat == "Edge Detection Filters" and default_val == "Canny Edge Detector")
        show_trans = cat == "Intensity Transforms & Enhancements"
        show_noise = cat == "Noise Models"
        show_color = (cat == "Color Space & Transforms" and default_val == "RGB to Grayscale")

        p1_label = "Target Region Classes (Bins)"
        p1_min, p1_max, p1_step, p1_val = 3, 5, 1, 3

        if cat == "Edge Detection Filters" and default_val == "Canny Edge Detector":
            p1_label, p1_min, p1_max, p1_step, p1_val = "Low Hysteresis Threshold", 0, 255, 1, 50

        return {
            sub_op_dropdown: gr.update(choices=sub_ops, value=default_val, label=f"{cat} Options"),
            kernel_slider: gr.update(visible=show_kernel),
            param1_slider: gr.update(label=p1_label, minimum=p1_min, maximum=p1_max, step=p1_step, value=p1_val,
                                     visible=show_param1),
            param2_slider: gr.update(visible=show_param2),
            transform_slider: gr.update(visible=show_trans),
            noise_slider: gr.update(visible=show_noise),
            color_slider: gr.update(visible=show_color)
        }


    category_dropdown.change(
        on_category_change,
        inputs=[category_dropdown],
        outputs=[sub_op_dropdown, kernel_slider, param1_slider, param2_slider, transform_slider, noise_slider,
                 color_slider]
    )


    def on_sub_op_change(cat, sub_op):
        show_param1, show_param2, show_color = False, False, False
        p1_label, p1_min, p1_max, p1_step, p1_val = "Parameter 1", 0, 100, 1, 1

        if cat == "Edge Detection Filters":
            show_param1 = True
            if sub_op == "Canny Edge Detector":
                p1_label, p1_min, p1_max, p1_step, p1_val = "Low Hysteresis Threshold", 0, 255, 1, 50
                show_param2 = True
            elif sub_op == "Laplacian of Gaussian (LoG)":
                p1_label, p1_min, p1_max, p1_step, p1_val = "Zero-Crossing Amplitude Threshold", 1, 50, 1, 10

        elif cat == "Segmentation & Clustered Bins":
            show_param1 = True
            if sub_op == "Multi-Otsu":
                p1_label, p1_min, p1_max, p1_step, p1_val = "Target Region Classes (Bins)", 3, 5, 1, 3
            elif sub_op == "Global Thresholding":
                p1_label, p1_min, p1_max, p1_step, p1_val = "Binary Threshold Break-point", 0, 255, 1, 127
            elif sub_op in ["Adaptive Gaussian", "Adaptive Mean"]:
                p1_label, p1_min, p1_max, p1_step, p1_val = "Neighborhood Local Block Size", 3, 99, 2, 11
            elif sub_op == "K-Means Clustering":
                p1_label, p1_min, p1_max, p1_step, p1_val = "Color Cluster K-Sets", 2, 10, 1, 3
            else:
                show_param1 = False

        elif cat == "Color Space & Transforms":
            show_color = (sub_op == "Luminance-Weighted Grayscale")

        return {
            param1_slider: gr.update(label=p1_label, minimum=p1_min, maximum=p1_max, step=p1_step, value=p1_val,
                                     visible=show_param1),
            param2_slider: gr.update(visible=show_param2),
            color_slider: gr.update(visible=show_color)
        }


    sub_op_dropdown.change(
        on_sub_op_change,
        inputs=[category_dropdown, sub_op_dropdown],
        outputs=[param1_slider, param2_slider, color_slider]
    )

    submit_btn.click(
        core_routing_pipeline,
        inputs=[source_image, category_dropdown, sub_op_dropdown, kernel_slider, param1_slider, param2_slider,
                transform_slider, noise_slider, color_slider],
        outputs=[output_image, histogram_output]
    )

if __name__ == "__main__":
    demo.launch()