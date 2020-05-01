// Your code will go here
// open up your console - if everything loaded properly you should see 0.9.0
let faceapi;
let video;
let detections;
let attentionDiv = document.getElementById("attention")
// by default all options are set to true
const detection_options = {
  withLandmarks: true,
  withDescriptors: false,
};

async function setup() {
  createCanvas(360, 270);

  // load up your video
  video = createCapture(VIDEO);
  video.size(width, height);
  // video.hide(); // Hide the video element, and just show the canvas
  faceapi = ml5.faceApi(video, detection_options, modelReady);
  textAlign(RIGHT);

  model = await tf.loadLayersModel(
    "https://motoitanigaki.github.io/model/model.json"
  );
  console.log(tf);
  console.log(model);
}

function modelReady() {
  console.log("ready!");
  console.log(faceapi);
  faceapi.detect(gotResults);
}

function gotResults(err, result) {
  if (err) {
    console.log(err);
    return;
  }
  // console.log(result)
  detections = result;

  // background(220);
  background(255);
  image(video, 0, 0, width, height);
  if (detections) {
    if (detections.length > 0) {
      // console.log(detections)
      drawBox(detections);
      drawLandmarks(detections);
    }
  }
  faceapi.detect(gotResults);
}

function drawBox(detections) {
  for (let i = 0; i < detections.length; i++) {
    const alignedRect = detections[i].alignedRect;
    const x = alignedRect._box._x;
    const y = alignedRect._box._y;
    const boxWidth = alignedRect._box._width;
    const boxHeight = alignedRect._box._height;

    noFill();
    stroke(161, 95, 251);
    strokeWeight(2);
    rect(x, y, boxWidth, boxHeight);
  }
}

async function drawLandmarks(detections) {
  noFill();
  stroke(161, 95, 251);
  strokeWeight(2);

  for (let i = 0; i < detections.length; i++) {
    // console.log(detections[i]);
    let faceFeaturesTmp = [];

    let centerPosition = [
      detections[i].alignedRect._box._x +
        detections[i].alignedRect._box.width / 2,
      detections[i].alignedRect._box._y +
        detections[i].alignedRect._box.height / 2,
    ];
    detections[i].landmarks._positions.forEach((object) => {
      faceFeaturesTmp.push([
        (Object.values(object)[0] - centerPosition[0]) /
          detections[i].alignedRect._box.width,
        Object.values(object)[1] -
          centerPosition[1] / detections[i].alignedRect._box.height,
      ]);
    });
    faceFeaturesTmp.push(
      Object.values([
        detections[i].alignedRect._box._height /
          detections[i].alignedRect._box._width,
      ])
    );

    // console.log(faceFeaturesTmp.flat());
    let prediction = await model
      .predict(tf.tensor([faceFeaturesTmp.flat()]))
      .data();
    let result = Array.from(prediction).map(function (p, i) {
      return {
        probability: p,
        classNumber: i,
      };
    });
    console.log(result[0], result[0].probability, result[0].classNumber);
    attentionDiv.innerText = `Attention : ${result[0].probability}

    const mouth = detections[i].parts.mouth;
    const nose = detections[i].parts.nose;
    const leftEye = detections[i].parts.leftEye;
    const rightEye = detections[i].parts.rightEye;
    const rightEyeBrow = detections[i].parts.rightEyeBrow;
    const leftEyeBrow = detections[i].parts.leftEyeBrow;

    drawPart(mouth, true);
    drawPart(nose, false);
    drawPart(leftEye, true);
    drawPart(leftEyeBrow, false);
    drawPart(rightEye, true);
    drawPart(rightEyeBrow, false);
  }
}

function drawPart(feature, closed) {
  beginShape();
  for (let i = 0; i < feature.length; i++) {
    const x = feature[i]._x;
    const y = feature[i]._y;
    vertex(x, y);
  }

  if (closed === true) {
    endShape(CLOSE);
  } else {
    endShape();
  }
}
