document.addEventListener('DOMContentLoaded', () => {
    // Handle the "Clear All History" button
    const clearHistoryButton = document.getElementById('clear-history');
    if (clearHistoryButton) {
        clearHistoryButton.addEventListener('click', async function() {
            const confirmed = confirm("Are you sure you want to delete all history? This action cannot be undone.");
            if (confirmed) {
                try {
                    const response = await fetch('/clear-history', {
                        method: 'POST',
                    });
                    if (response.ok) {
                        window.location.reload(); // Refresh the page
                    } else {
                        alert('Failed to clear history.');
                    }
                } catch (error) {
                    alert('Error clearing history: ' + error.message);
                }
            }
        });
    }

    // Handle the "Back to List" button
    const backToListButton = document.getElementById('back-to-list');
    if (backToListButton) {
        backToListButton.addEventListener('click', function() {
            document.getElementById('history-list').style.display = 'block';
            document.getElementById('history-detail').style.display = 'none';
        });
    }

    // Handle the "View Details" button
    document.querySelectorAll('.view-detail-button').forEach(button => {
        button.addEventListener('click', function() {
            const analysisId = this.dataset.id;
            const originalImage = this.dataset.original;
            const segmentedImage = this.dataset.segmented;
            const gradcamImage = this.dataset.gradcam;

            // Update the detail section with the selected analysis data
            document.getElementById('history-original-image').src = originalImage;
            document.getElementById('history-segmented-image').src = segmentedImage;
            document.getElementById('history-gradcam-image').src = gradcamImage;

            // Show the detail section and hide the list
            document.getElementById('history-list').style.display = 'none';
            document.getElementById('history-detail').style.display = 'block';
        });
    });
});