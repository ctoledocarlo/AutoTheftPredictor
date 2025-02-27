const form = document.getElementById('prediction-form');
const resultDiv = document.getElementById('prediction-result');
const jsonResponse = document.getElementById('jsonResponse')
const jsonBody = document.getElementById('jsonBody')

form.addEventListener('submit', (event) => {
    event.preventDefault();

    // Get the data from the user input
    const formData = new FormData(form);
    const data = [{}]; // Initialize as an array to match API expectations

    // Get the date value and split it into year, month, and day
    const date = new Date(formData.get('date'));
    data[0].REPORT_YEAR = date.getFullYear();
    data[0].REPORT_MONTH = date.getMonth() + 1; // Months are 0-indexed, so add 1
    data[0].REPORT_DAY = date.getDate();

    // Get the hour value and convert it to a 24-hour format
    let hour = parseInt(formData.get('HOUR_VALUE'));
    const hourPeriod = formData.get('HOUR_PERIOD');

    if (hourPeriod === 'PM' && hour !== 12) {
        hour += 12; // Convert PM hours (except 12) to 24-hour format
    } else if (hourPeriod === 'AM' && hour === 12) {
        hour = 0; // Convert 12 AM to 0 hour
    }

    // Calculate sine and cosine of the hour
    data[0].REPORT_HOUR_sin = Math.sin((hour % 24) * (Math.PI / 12)); // Normalize hour for sine
    data[0].REPORT_HOUR_cos = Math.cos((hour % 24) * (Math.PI / 12)); // Normalize hour for cosine

    // Get latitude and longitude
    data[0].LAT_WGS84 = parseFloat(formData.get('LAT_WGS84'));
    data[0].LONG_WGS84 = parseFloat(formData.get('LONG_WGS84'));

    // Post request to the API
    fetch('http://127.0.0.1:5000/predictrisk', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
        .then(response => response.json())
        .then(prediction => {
            // Display the result in the client

            jsonBody.textContent = `JSON Body: \n${JSON.stringify(data)}`;
            jsonResponse.textContent = `JSON Response: \n${JSON.stringify(prediction)}`;
            
            if (prediction.predictions && prediction.predictions[0] === 1) {
                resultDiv.textContent = 'The car is likely to be stolen.';
            } else {
                resultDiv.textContent = 'The car is not likely to be stolen.';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultDiv.textContent = 'An error occurred.';
        });
});
