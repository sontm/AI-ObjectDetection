// Import dependencies
import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocossd from "@tensorflow-models/coco-ssd";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as knnClassifier from'@tensorflow-models/knn-classifier';

import Webcam from "react-webcam";
import "./App.css";
import { drawRect } from "./utilities";

function App() {
  let classifier = null;
  let mobilenet = null;
  const maskImageCount = 5;
  const noMaskImageCount = 5;

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const canvasTrainRef = useRef(null);

  // Main function
  const runCoco = async () => {
    const net = await cocossd.load();
    console.log("Handpose model loaded.");
    //  Loop and detect hands
    setInterval(() => {
      detect(net);
    }, 100);
  };

  const detect = async (net) => {
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Make Detections
      const obj = await net.detect(video);
      let tfTestImage = tf.browser.fromPixels(video)
      const xlogits = mobilenet.infer(tfTestImage, 'conv_preds');
      const prediction = await classifier.predictClass(xlogits);
      // Draw mesh
      const ctx = canvasRef.current.getContext("2d");
      drawRect(obj, ctx, prediction); 
    }
  };

  //useEffect(()=>{runCoco()},[]);
  useEffect(async ()=>{
    // Load mobilenet.
    mobilenet = await mobilenetModule.load({version: 2, alpha: 1});
    // Create the classifier.
    classifier = knnClassifier.create();

    // var canvasTrainRef = document.createElement('CANVAS');
    let canvasHeight = canvasTrainRef.current.height;
    let canvasWidth = canvasTrainRef.current.width;

    // canvasTrainRef.setAttribute('width',canvasWidth);
    // canvasTrainRef.setAttribute('height',canvasHeight);

    var ctx = canvasTrainRef.current.getContext('2d');

    const maskImages = document.querySelectorAll('.train-img-mask');
    maskImages.forEach((img) => {
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      // ctx.drawImage(img,0,0, img.width,    img.height,
      //     0,0, canvasWidth, canvasHeight);
      ctx.drawImage(img,0,0, canvasWidth, canvasHeight);
      const tfImg = tf.browser.fromPixels(ctx.getImageData(0, 0, canvasWidth, canvasHeight));
      const logits = mobilenet.infer(tfImg, 'conv_preds');
      classifier.addExample(logits, 1); // has mask
    });
    
    const nomaskImages = document.querySelectorAll('.train-img-nomask');
    nomaskImages.forEach(img => {
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);
      ctx.drawImage(img,0,0, canvasWidth, canvasHeight);
      const tfImg = tf.browser.fromPixels(ctx.getImageData(0, 0, canvasWidth, canvasHeight));
      const logits = mobilenet.infer(tfImg, 'conv_preds');
      classifier.addExample(logits, 0); // no mask
    });

    // const testImage = document.getElementById('test-nomask');
    // ctx.clearRect(0, 0, canvasTrainRef.current.width, canvasTrainRef.current.height);
    // ctx.drawImage(testImage,0,0);
    // const tfTestImage = tf.browser.fromPixels(ctx.getImageData(0, 0, canvasTrainRef.current.width, canvasTrainRef.current.height));
    // const xlogits = mobilenet.infer(tfTestImage, 'conv_preds');
    // const prediction = await classifier.predictClass(xlogits);
    // console.log(prediction.confidences)
    // if (prediction.label == 1) { // no mask - red border
    //   console.log('no-mask');
    // } else { // has mask - green border
    //   console.log('mask');
    // }

    runCoco()
  },[]);

  let viewMask = []
  for (let i = 1; i <= maskImageCount; i++) {
    viewMask.push(<img crossorigin="anonymous" className="train-img-mask" src={"/data/images/mask/" + i + ".jpg"} style={{height:"100%", width: "auto"}}/>)
  }
  let viewNoMask = []
  for (let i = 1; i <= noMaskImageCount; i++) {
    viewNoMask.push(<img crossorigin="anonymous"  className="train-img-nomask" src={"/data/images/no_mask/" + i + ".jpg"} style={{height:"100%", width: "auto"}}/>)
  }
  
  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          muted={true} 
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 8,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasTrainRef}
          style={{
            position: "absolute",
            marginRight: "0",
            right: 0,
            textAlign: "center",
            zindex: 10,
            width: 640,
            height: 480,
            display: "none"
          }}
        />
      </header>

      <div style={{height: "100px"}}>
        {viewMask}
      </div>
      <div style={{height: "100px"}}>
        {viewNoMask}
      </div>

      <div style={{height: "100px"}}>
        <img id="test-mask" crossorigin="anonymous" src={"/data/images/mask.jpg"} style={{height:"100%", width: "auto"}}/>
      </div>
      <div style={{height: "100px"}}>
        <img id="test-nomask" crossorigin="anonymous" src={"/data/images/no_mask.jpg"} style={{height:"100%", width: "auto"}}/>
      </div>
    </div>
  );
}

export default App;
