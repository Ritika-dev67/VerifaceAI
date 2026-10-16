async function predictImage() {

    const input = document.getElementById("imageInput");

    if (input.files.length === 0) {

        alert("Please select an image");

        return;
    }

    const file = input.files[0];

    const formData = new FormData();

    formData.append("image", file);


    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        document.getElementById("result").innerText =
            `Prediction: ${data.prediction.toUpperCase()}
             Confidence: ${data.confidence}%`;

    }

    catch (error) {

        console.error(error);

        document.getElementById("result").innerText =
            "Error connecting to backend";
    }
}

async function predictVideo() {

    const input =
        document.getElementById("videoInput");

    const result =
        document.getElementById("result");

    if (!input.files.length) {

        result.innerText =
            "Please select a video.";

        return;
    }

    const video =
        input.files[0];

    const formData =
        new FormData();

    formData.append(
        "video",
        video
    );

    result.innerText =
        "Analyzing video...";

    try {

        const response =
            await fetch(
                "http://127.0.0.1:5000/predict-video",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (data.error) {

            result.innerText =
                data.error;

            return;
        }

        result.innerHTML = `
            <h2>Result: ${data.prediction}</h2>

            <p>
                Confidence:
                ${data.confidence}%
            </p>

            <p>
                Fake Frames:
                ${data.fake_frames}
            </p>

            <p>
                Real Frames:
                ${data.real_frames}
            </p>
        `;

    } catch (error) {

        console.error(error);

        result.innerText =
            "Something went wrong.";
    }
}
if (data.error) {
    document.getElementById("result").innerText = data.error;
    return;
}