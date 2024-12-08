const cognitiveTestForm = document.getElementById('cognitive-test');
const resultDiv = document.getElementById('result');

cognitiveTestForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const files = [];
    for (let i = 1; i <= 3; i++) {
        const fileInput = document.getElementById(`file${i}`);
        files.push(fileInput.files[0]);
    }
    // Send files to server for prediction
    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify(files),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then((data) => {
        const result = data.prediction;
        resultDiv.innerText = `Based on your cognitive test results, you are at risk of ${result}.`;
    })
    .catch((error) => {
        console.error(error);
    });
});