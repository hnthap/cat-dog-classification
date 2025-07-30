const sharp = require('sharp');


function main() {
    if (process.argv.length < 3) {
        console.error('No image specified!');
        return;
    }
    const imagePath = process.argv[2];
    infer(imagePath).then(showPrediction).catch(console.error);
}

async function infer(imagePath) {
    const body = {
        inputs: [{ 
            name: 'images',
            shape: [1, 3, 224, 224],
            datatype: 'FP32',
            data: await preprocess(imagePath)
        }]
    };
    const res = await fetch('http://localhost:8000/v2/models/catdog/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    return await res.json();
}

async function preprocess(imagePath) {
    const image = sharp(imagePath);
    const { width, height } = await image.metadata();
    const scale = 256 / Math.min(width, height);
    const newWidth = Math.round(width * scale);
    const newHeight = Math.round(height * scale);
    const buffer = await image
        .resize(newWidth, newHeight, { fit: 'outside' })
        .extract({
            left: Math.round((newWidth - 224) / 2),
            top: Math.round((newHeight - 224) / 2),
            width: 224,
            height: 224
        })
        .raw()
        .toBuffer({ resolveWithObject: true });
    const { data, info } = buffer;
    const pixels = new Float32Array(data.length);
    const means = [0.485, 0.456, 0.406];
    const stds = [0.229, 0.224, 0.225];
    const redStart = 0;
    const greenStart = 1 * 224 * 224;
    const blueStart = 2 * 224 * 224;
    for (let y = 0; y < 224; y += 1) {
        for (let x = 0; x < 224; x++) {
            const hwcIndex = (y * 224 + x) * 3;
            const baseChwIndex = y * 224 + x;
            pixels[redStart + baseChwIndex] = ((data[hwcIndex] / 255.0) - means[0]) / stds[0];
            pixels[greenStart + baseChwIndex] = ((data[hwcIndex + 1] / 255.0) - means[1]) / stds[1];
            pixels[blueStart + baseChwIndex] = ((data[hwcIndex + 2] / 255.0) - means[2]) / stds[2];
        }
    }
    return Array.from(pixels);
}

function showPrediction(output) {
    const [dog, cat] = output.outputs[0].data;
    console.log('Dog:', Math.round(dog * 1000) / 10, '%');
    console.log('Cat:', Math.round(cat * 1000) / 10, '%');
}

main();
