export const drawRect = (detections, ctx, predictionMask) =>{
  // Loop through each prediction
  let strMask = "";
  if (predictionMask.label == 1) {
    strMask = "MASK";
  } else {
    strMask = "NOMASK";
  }
  detections.forEach(prediction => {

    // Extract boxes and classes
    const [x, y, width, height] = prediction['bbox']; 
    const text = prediction['class'] + "_" + strMask; 

    // Set styling
    const color = Math.floor(Math.random()*16777215).toString(16);
    ctx.strokeStyle = '#' + color
    ctx.font = '18px Arial';

    // Draw rectangles and text
    ctx.beginPath();   
    ctx.fillStyle = '#' + color
    ctx.fillText(text, x, y);
    ctx.rect(x, y, width, height); 
    ctx.stroke();
  });
}
