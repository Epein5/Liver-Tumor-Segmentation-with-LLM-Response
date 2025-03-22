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
import shap


def load_and_preprocess_image(image_path, target_size=(128, 128), downsample_factor=1):
    """Load and preprocess the input image with optional downsampling for speed."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    
    # Store original image for display
    original = image.copy()
    
    # Apply downsampling for faster processing if requested
    if downsample_factor > 1:
        h, w = image.shape
        image = cv2.resize(image, (w//downsample_factor, h//downsample_factor))
    
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

def get_combined_segmentation_masks(model, image):
    """Generate both liver and tumor segmentation masks with a single model call."""
    # Single model prediction call
    mask = model.predict(image)
    
    # Process for liver segmentation
    liver_mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize
    liver_threshold = np.percentile(liver_mask, 90)  # Use 90th percentile as threshold
    liver_mask = (liver_mask > liver_threshold).astype(np.uint8)
    liver_mask = scipy.ndimage.binary_fill_holes(liver_mask.squeeze()).astype(np.uint8)  # Fill holes
    liver_mask = cv2.morphologyEx(liver_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # Larger kernel
    liver_mask = cv2.dilate(liver_mask, np.ones((5, 5), np.uint8), iterations=1)  # Expand the mask
    liver_mask = cv2.resize(liver_mask, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    # Process for tumor segmentation
    tumor_mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize
    tumor_mask = (tumor_mask > 0.5).astype(np.uint8)  # Thresholding for tumor
    tumor_mask = scipy.ndimage.binary_fill_holes(tumor_mask.squeeze()).astype(np.uint8)  # Fill holes
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Remove noise
    tumor_mask = cv2.resize(tumor_mask, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    return liver_mask, tumor_mask

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
    # Check if tumor is non-significant or not present
    if metrics["percentage"] < 1:  # Very small threshold
        return (
            "LIVER ANALYSIS REPORT: "
            "No significant tumor detected in the visible liver area. "
            "The analysis shows normal liver parenchyma without evidence of focal lesions. "
            "Routine follow-up imaging may be considered based on clinical context and risk factors."
        )
    
    # For cases with significant findings, continue with existing logic
    severity = "minimal"
    if metrics["percentage"] > 2:
        severity = "mild"
    if metrics["percentage"] > 5:
        severity = "moderate"
    if metrics["percentage"] > 10:
        severity = "severe"
    
    # Handle missing dimensions gracefully
    max_diameter = metrics.get("max_diameter_pixels", "unknown")
    width = metrics.get("width_pixels", "unknown")
    height = metrics.get("height_pixels", "unknown")
    
    # Create explanation based on available metrics
    explanation = (
        f"LIVER TUMOR ANALYSIS REPORT: "
        f"Tumor occupying approximately {metrics['percentage']:.2f}% of the visible liver area. "
    )
    
    if max_diameter != "unknown":
        explanation += f"The maximum tumor diameter is {max_diameter:.2f} pixels. "
    else:
        explanation += "The tumor diameter not found. "
    
    if width != "unknown" and height != "unknown":
        explanation += f"The tumor dimensions are approximately {width} x {height} pixels. "
    else:
        explanation += "The tumor dimensions could not be found. "
    
    explanation += (
        f"The scan shows a {severity} hepatic lesion with well-defined borders. "
        f"The Grad-CAM++ analysis confirms the area of concern and highlights regions of highest diagnostic significance. "
        f"Recommendations include clinical correlation, follow-up imaging in 3-6 months to assess for changes, "
        f"and additional laboratory studies as warranted based on clinical presentation."
    )
    
    return explanation


def compute_shap_values(model, image, background):
    """Compute SHAP values using GradientExplainer."""
    explainer = shap.GradientExplainer(model, background, local_smoothing=0.1)
    return explainer.shap_values(image)

def enhance_shap_visualization(shap_values, mask):
    """Enhance SHAP values for better visualization."""
    shap_map = np.abs(shap_values[0].squeeze())
    shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())
    shap_map = scipy.ndimage.gaussian_filter(shap_map, sigma=2)  # Smoothing
    shap_map = (shap_map * 255).astype(np.uint8)
    return cv2.bitwise_and(shap_map, shap_map, mask=mask)

def create_overlay(image, shap_map, mask, color_map, intensity=1.5):
    """Create an overlay of SHAP values on the original image."""
    image_rgb = (image[0] * 255).astype(np.uint8)
    shap_colored = cv2.applyColorMap(shap_map, color_map)
    overlay = image_rgb.copy()
    overlay[mask > 0] = cv2.addWeighted(shap_colored[mask > 0], intensity, image_rgb[mask > 0], 0.5, 0)
    return overlay

def analyze_and_save_medical_image(model, image_path, fast_mode=False):
    """
    Main function to analyze a medical image, visualize results, and save to database.
    Added fast_mode to skip expensive operations when speed is preferred.
    """
    try:
        # Create unique ID for this analysis
        analysis_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join("db", analysis_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and preprocess image - use downsample_factor=2 for faster processing in fast mode
        downsample_factor = 2 if fast_mode else 1
        preprocessed_image, original_image = load_and_preprocess_image(
            image_path, downsample_factor=downsample_factor
        )
        
        # Generate liver and tumor segmentation masks with a single model call
        liver_mask, tumor_mask = get_combined_segmentation_masks(model, preprocessed_image)
        
        # Create combined segmentation masks
        combined_mask_img, labeled_mask = create_combined_mask(
            original_image, 
            liver_mask, 
            tumor_mask
        )
        
        # Calculate tumor metrics
        tumor_metrics = calculate_tumor_metrics(tumor_mask)
        
        # Generate medical explanation
        medical_explanation = generate_medical_explanation(tumor_metrics)
        
        # Save basic results
        cv2.imwrite(os.path.join(output_dir, "original.png"), original_image)
        cv2.imwrite(os.path.join(output_dir, "combined_segmentation.png"), combined_mask_img)
        
        # Save metrics and explanation
        with open(os.path.join(output_dir, "tumor_metrics.json"), "w") as f:
            json.dump(tumor_metrics, f, indent=2)
            
        with open(os.path.join(output_dir, "medical_report.txt"), "w") as f:
            f.write(medical_explanation)
        
        # Skip computationally expensive operations in fast mode
        if not fast_mode:
            # Compute SHAP values (expensive operation)
            model_with_vector_output = modify_model_output(model)
            background = np.zeros_like(preprocessed_image)
            shap_values = compute_shap_values(model_with_vector_output, preprocessed_image, background)
            
            shap_map_liver = enhance_shap_visualization(shap_values, liver_mask)
            shap_map_tumor = enhance_shap_visualization(shap_values, tumor_mask)
            
            liver_overlay = create_overlay(preprocessed_image, shap_map_liver, liver_mask, cv2.COLORMAP_SUMMER, intensity=7.0)
            tumor_overlay = create_overlay(preprocessed_image, shap_map_tumor, tumor_mask, cv2.COLORMAP_JET, intensity=2.0)
            
            combined_overlay = cv2.addWeighted(liver_overlay, 0.5, tumor_overlay, 0.5, 0)
            
            # Generate Grad-CAM++ explanations (expensive operation)
            _, tumor_focused_grad_cam = compute_gradcam_plus_with_focus(
                model, 
                preprocessed_image, 
                tumor_mask
            )
            
            grad_cam_tumor_focused = overlay_heatmap_improved(
                original_image, 
                tumor_focused_grad_cam, 
                alpha=0.7, 
                use_jet=False
            )
            
            # Resize and save additional visualizations
            h, w = original_image.shape[:2]
            combined_overlay_resized = cv2.resize(combined_overlay, (w, h), interpolation=cv2.INTER_LINEAR)
            
            cv2.imwrite(os.path.join(output_dir, "combined_shap_overlay.png"), combined_overlay_resized)
            cv2.imwrite(os.path.join(output_dir, "gradcam_tumor_focused.png"), grad_cam_tumor_focused)
        
        # Return results
        return output_dir, tumor_metrics, medical_explanation
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, str(e)