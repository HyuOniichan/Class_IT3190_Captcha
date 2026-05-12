document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const predictButton = document.getElementById('predictButton');
    const uploadedImage = document.getElementById('uploadedImage');
    const predictionText = document.getElementById('predictionText');

    let selectedFile = null;

    imageUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => {
                uploadedImage.src = e.target.result;
                uploadedImage.style.display = 'block'; // Show the image
                predictionText.textContent = ''; // Clear previous prediction
            };
            reader.readAsDataURL(selectedFile);
        } else {
            uploadedImage.src = '#';
            uploadedImage.style.display = 'none';
        }
    });

    predictButton.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('Please upload an image first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(
                    errorData.detail ||
                    'Something went wrong during prediction.'
                );
            }

            const data = await response.json();
            predictionText.textContent = data.prediction;
        } catch (error) {
            console.error('Error:', error);
            predictionText.textContent = `Error: ${error.message}`;
            alert(`Error during prediction: ${error.message}`);
        }
    });
});
