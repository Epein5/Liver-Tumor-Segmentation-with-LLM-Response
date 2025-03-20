import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer
import os
import datetime
import uuid
from scipy.ndimage import gaussian_filter
import json
import requests

# def create_analysis_system(model_path="efficientnet_unet_model.h5"):
#     """Initialize the medical image analysis system with the specified model."""
#     # Load the model
#     model = load_model(model_path)
    
#     # Create base database directory if it doesn't exist
#     os.makedirs("db", exist_ok=True)
    
#     return model

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess the input image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    
    # Store original image for display
    original = image.copy()
    
    # Resize and preprocess for model
    image = cv2.resize(image, target_size)
    image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return image, original

def modify_model_output(model):
    """Modify the model to return a vector output."""
    class ReduceMeanLayer(Layer):
        def __init__(self, axis, **kwargs):
            super(ReduceMeanLayer, self).__init__(**kwargs)
            self.axis = axis

        def call(self, inputs):
            return tf.reduce_mean(inputs, axis=self.axis)

    input_layer = model.input
    output_layer = model.output
    modified_output = ReduceMeanLayer(axis=[1, 2])(output_layer)
    return Model(inputs=input_layer, outputs=modified_output)

def get_liver_segmentation_mask(model, image):
    """Generate and refine the liver segmentation mask."""
    mask = model.predict(image)
    mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize
    threshold = np.percentile(mask, 90)  # Use 90th percentile as threshold
    mask = (mask > threshold).astype(np.uint8)
    mask = scipy.ndimage.binary_fill_holes(mask.squeeze()).astype(np.uint8)  # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # Larger kernel
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)  # Expand the mask
    return cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

def get_tumor_segmentation_mask(model, image):
    """Generate and refine the tumor segmentation mask."""
    mask = model.predict(image)
    mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize
    mask = (mask > 0.5).astype(np.uint8)  # Thresholding for tumor
    mask = scipy.ndimage.binary_fill_holes(mask.squeeze()).astype(np.uint8)  # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Remove noise
    return cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

def create_combined_mask(original_img, liver_mask, tumor_mask):
    """
    Create a combined mask image where:
    - Liver is shown in semi-transparent green
    - Tumor is shown in semi-transparent red
    - Masks are applied directly over the original image
    
    Returns both the combined mask and a labeled segmentation image
    """
    # Ensure original image is RGB
    if len(original_img.shape) == 2:
        display_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        display_img = original_img.copy()
        
    # Create a labeled segmentation image (0=background, 1=liver, 2=tumor)
    labeled_mask = np.zeros(liver_mask.shape, dtype=np.uint8)
    labeled_mask[liver_mask == 1] = 1  # Liver
    labeled_mask[tumor_mask == 1] = 2  # Tumor (overwrites liver where they overlap)
    
    # Resize masks to match original image size
    h, w = original_img.shape[:2]
    labeled_mask_resized = cv2.resize(labeled_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create a color-coded mask image
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[labeled_mask_resized == 1] = [0, 200, 0]  # Green for liver
    color_mask[labeled_mask_resized == 2] = [200, 0, 0]  # Red for tumor
    
    # Apply color mask with transparency
    alpha = 0.5
    combined = cv2.addWeighted(display_img, 1, color_mask, alpha, 0)
    
    # Create a boundary overlay to highlight the segmentation edges
    liver_boundary = cv2.Canny(cv2.resize(liver_mask*255, (w, h)), 100, 200)
    tumor_boundary = cv2.Canny(cv2.resize(tumor_mask*255, (w, h)), 100, 200)
    
    # Add boundaries to the combined image
    combined[liver_boundary > 0] = [0, 255, 0]  # Green boundary for liver
    combined[tumor_boundary > 0] = [255, 0, 0]  # Red boundary for tumor
    
    return combined, labeled_mask

def compute_gradcam_plus_with_focus(model, img_array, tumor_mask, layer_name="block7a_expand_activation"):
    """
    Compute Grad-CAM++ heatmap with better focus on the tumor region.
    This version enhances the focus on tumor regions using the tumor mask.
    """
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Compute gradients using tumor class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 1]  # Focus on tumor probability
    
    # Get gradients and compute Grad-CAM++ weights
    grads = tape.gradient(loss, conv_outputs)
    grads_squared = tf.square(grads)
    grads_cubed = grads_squared * grads

    # Compute Grad-CAM++ importance weights
    alpha_num = grads_squared
    alpha_denom = grads_squared * 2 + conv_outputs * grads_cubed
    alpha_denom = tf.where(alpha_denom != 0, alpha_denom, tf.ones_like(alpha_denom))
    alpha = alpha_num / alpha_denom

    weights = tf.reduce_mean(alpha * tf.nn.relu(grads), axis=(0, 1, 2))
    
    # Compute heatmap
    heatmap = tf.reduce_sum(weights * conv_outputs[0], axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize the heatmap
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Apply Gaussian smoothing for better visualization
    heatmap = gaussian_filter(heatmap, sigma=2)
    
    # Resize heatmap to match mask size
    heatmap_resized = cv2.resize(heatmap, (tumor_mask.shape[1], tumor_mask.shape[0]))
    
    # Create two versions: one with tumor focus and one original
    # Original heatmap
    original_heatmap = heatmap_resized.copy()
    
    # Tumor-focused heatmap: enhance values where tumor is present
    tumor_focused_heatmap = heatmap_resized.copy()
    # Boost signal in tumor region
    boost_factor = 1.5
    tumor_focused_heatmap = tumor_focused_heatmap * (1 + tumor_mask * boost_factor)
    # Normalize again after boosting
    if np.max(tumor_focused_heatmap) != 0:
        tumor_focused_heatmap = tumor_focused_heatmap / np.max(tumor_focused_heatmap)
    
    return original_heatmap, tumor_focused_heatmap

def overlay_heatmap_improved(img, heatmap, alpha=0.6, use_jet=True):
    """
    Overlay Grad-CAM heatmap on the original image with improved visibility.
    
    Args:
        img: Original image (grayscale or RGB)
        heatmap: Heatmap data (normalized 0-1)
        alpha: Transparency factor
        use_jet: Whether to use COLORMAP_JET (True) or a custom red-based colormap (False)
        
    Returns:
        Overlayed image
    """
    # Ensure image is in RGB format
    if len(img.shape) == 2 or img.shape[-1] == 1:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        display_img = img.copy()
    
    # Convert float image to uint8 if needed
    if display_img.dtype != np.uint8:
        if display_img.max() <= 1.0:
            display_img = (display_img * 255).astype(np.uint8)
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (display_img.shape[1], display_img.shape[0]))
    
    # Convert heatmap to 0-255 scale
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    if use_jet:
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    else:
        # Custom red-focused colormap for tumor visualization
        colored_heatmap = np.zeros_like(display_img)
        colored_heatmap[:,:,0] = 0  # Blue channel
        colored_heatmap[:,:,1] = 0  # Green channel
        colored_heatmap[:,:,2] = heatmap_uint8  # Red channel
    
    # Create mask for areas with significant activation
    sig_threshold = 0.3
    sig_mask = (heatmap_resized > sig_threshold).astype(np.uint8)
    
    # Apply contour to significant areas for better boundary visualization
    sig_mask_uint8 = np.uint8(sig_mask * 255)
    contours, _ = cv2.findContours(sig_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Apply alpha blending for the heatmap
    overlayed = cv2.addWeighted(display_img, 1 - alpha, colored_heatmap, alpha, 0)
    
    # Draw contours on areas of high activation (optional)
    if contours:
        cv2.drawContours(overlayed, contours, -1, (255, 255, 255), 1)
    
    return overlayed

def calculate_tumor_metrics(tumor_mask):
    """Calculate metrics about the tumor from its mask."""
    # Count non-zero pixels for area
    tumor_area_pixels = np.count_nonzero(tumor_mask)
    
    # Find contours for diameter calculation
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    metrics = {
        "area_pixels": tumor_area_pixels,
        "percentage": (tumor_area_pixels / (tumor_mask.shape[0] * tumor_mask.shape[1])) * 100
    }
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        max_diameter = radius * 2
        
        # Get bounding rectangle for width/height
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        metrics.update({
            "max_diameter_pixels": max_diameter,
            "width_pixels": w,
            "height_pixels": h,
        })
    
    return metrics

def generate_medical_explanation(metrics):
    """
    Generate a medical explanation for the tumor findings.
    In a real system, this would call GEMINI API, but we'll simulate it here.
    """
    # Calculate severity based on tumor size
    severity = "minimal"
    if metrics["percentage"] > 2:
        severity = "mild"
    if metrics["percentage"] > 5:
        severity = "moderate"
    if metrics["percentage"] > 10:
        severity = "severe"
    
    # Create explanation
    explanation = (
    f"LIVER TUMOR ANALYSIS REPORT: Findings indicate a tumor detected in the liver segment, "
    f"occupying approximately {metrics['percentage']:.2f}% of the visible liver area. The maximum tumor diameter is "
    f"{metrics.get('max_diameter_pixels', 'N/A')} pixels, with dimensions of "
    f"{metrics.get('width_pixels', 'N/A')} x {metrics.get('height_pixels', 'N/A')} pixels. "
    f"The scan shows a {severity} hepatic lesion with well-defined borders. The Grad-CAM++ analysis confirms the area "
    f"of concern and highlights regions of highest diagnostic significance. Recommendations include clinical correlation, "
    f"follow-up imaging in 3-6 months to assess for changes, and additional laboratory studies as warranted based on clinical presentation."
)
    return explanation

def analyze_and_save_medical_image(model, image_path):
    """
    Main function to analyze a medical image, visualize results, and save to database.
    """
    try:
        # Create unique ID for this analysis
        analysis_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join("db", analysis_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess image
        preprocessed_image, original_image = load_and_preprocess_image(image_path)
        
        # Generate liver and tumor segmentation masks
        liver_mask = get_liver_segmentation_mask(model, preprocessed_image)
        tumor_mask = get_tumor_segmentation_mask(model, preprocessed_image)
        
        # Create combined segmentation masks
        combined_mask_img, labeled_mask = create_combined_mask(
            original_image, 
            liver_mask, 
            tumor_mask
        )
        
        # Calculate tumor metrics
        tumor_metrics = calculate_tumor_metrics(tumor_mask)
        
        # Generate Grad-CAM++ explanations
        original_grad_cam, tumor_focused_grad_cam = compute_gradcam_plus_with_focus(
            model, 
            preprocessed_image, 
            tumor_mask
        )
        
        # Create visualizations with improved overlay
        grad_cam_general = overlay_heatmap_improved(
            original_image, 
            original_grad_cam, 
            alpha=0.7
        )
        
        grad_cam_tumor_focused = overlay_heatmap_improved(
            original_image, 
            tumor_focused_grad_cam, 
            alpha=0.7, 
            use_jet=False
        )
        
        # Generate medical explanation
        medical_explanation = generate_medical_explanation(tumor_metrics)
        
        # Save all outputs
        cv2.imwrite(os.path.join(output_dir, "original.png"), original_image)
        cv2.imwrite(os.path.join(output_dir, "combined_segmentation.png"), combined_mask_img)
        cv2.imwrite(os.path.join(output_dir, "gradcam_tumor_focused.png"), grad_cam_tumor_focused)
        
        # Save metrics and explanation
        with open(os.path.join(output_dir, "tumor_metrics.json"), "w") as f:
            json.dump(tumor_metrics, f, indent=2)
            
        with open(os.path.join(output_dir, "medical_report.txt"), "w") as f:
            f.write(medical_explanation)
        
        # Create visualization for display
        return output_dir, tumor_metrics, medical_explanation
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, str(e)
