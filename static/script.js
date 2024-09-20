document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const startButton = document.getElementById('start-recognition');
    const resultDiv = document.getElementById('result');

    // Access webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => {
            console.error('Error accessing webcam: ', err);
        });

    startButton.addEventListener('click', function() {
        startRecognition();
    });

    function startRecognition() {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        canvas.width = video.width;
        canvas.height = video.height;

        setInterval(() => {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('video_frame', blob);

                fetch('/recognize_gesture', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    resultDiv.textContent = `Gesture Detected: ${data.gesture}`;
                })
                .catch(err => {
                    console.error('Error recognizing gesture: ', err);
                });
            }, 'image/jpeg');
        }, 1000); // Send frame every second
    }
});
