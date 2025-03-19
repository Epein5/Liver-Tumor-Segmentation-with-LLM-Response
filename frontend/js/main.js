document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('image-upload');
    const fileNameDisplay = document.getElementById('file-name');
    const previewImage = document.getElementById('preview-image');
    const uploadForm = document.getElementById('upload-form');
    const loadingIndicator = document.getElementById('loading');
    const resultsContainer = document.getElementById('results');
    const originalImage = document.getElementById('original-image');
    const segmentedImage = document.getElementById('segmented-image');
    const explanationText = document.getElementById('explanation-text');
    const tumorSize = document.getElementById('tumor-size');
    const confidence = document.getElementById('confidence');

    // Handle file input change
    fileInput.addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file) {
            fileNameDisplay.textContent = file.name;
            previewImage.src = URL.createObjectURL(file);
            previewImage.style.display = 'block';
        } else {
            fileNameDisplay.textContent = 'No file chosen';
            previewImage.style.display = 'none';
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', async function (event) {
        event.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            alert('Please select an image file first.');
            return;
        }

        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultsContainer.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/segment', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server responded with an error.');
            }

            const data = await response.json();
            console.log('Analysis completed:', data);

            // Fetch the result data using the result_id
            const resultResponse = await fetch(`/db/${data.result_id}.json`);
            if (!resultResponse.ok) {
                throw new Error('Failed to fetch analysis results.');
            }

            const resultData = await resultResponse.json();
            console.log('Result data:', resultData);

            // Display the results
            originalImage.src = resultData.original_image;
            segmentedImage.src = resultData.segmented_image;
            explanationText.textContent = resultData.medical_explanation;

            // Access tumor_metrics fields
            tumorSize.textContent = `${resultData.tumor_metrics.max_diameter_pixels.toFixed(2)} pixels`;
            confidence.textContent = `${resultData.tumor_metrics.percentage.toFixed(2)}%`;

            // Hide loading indicator and show results
            loadingIndicator.style.display = 'none';
            resultsContainer.style.display = 'block';

        } catch (error) {
            console.error('Error during analysis:', error);
            alert('An error occurred while processing the image. Please try again.');
            loadingIndicator.style.display = 'none';
        }
    });
});