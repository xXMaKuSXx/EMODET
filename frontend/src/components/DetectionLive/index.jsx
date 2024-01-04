import { useState } from "react";
import axios from "axios";
import Button from "../Button";
import EmotionDetector from "../EmotionDetector";
import styles from "@/styles/Card.module.css";

const DetectionLive = (model) => {
  const [isRecording, setIsRecording] = useState(false);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      const recorder = new MediaRecorder(stream);
      const response = await axios("http://localhost:8080/start-recording");

      console.log(response.data.text);
      recorder.start();

      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };
  const stopRecording = async () => {
    try {
      const response = await axios("http://localhost:8080/stop-recording");
      console.log(response.data.text);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      const recorder = new MediaRecorder(stream);
      setIsRecording(false);
      recorder.stop();
    } catch (error) {
      setIsRecording(false);
      console.error("Error stopping the recording:", error);
    }
  };

  return (
    <div className={`${styles.detectionresult}`}>
      {isRecording ? (
        <>
          <EmotionDetector model={model.model} />
          <Button onClick={stopRecording}>Stop Recording</Button>
        </>
      ) : (
        <>
          <h2>The Result will be shown over here!</h2>
          <Button onClick={startRecording}>Start Recording</Button>
        </>
      )}
    </div>
  );
};

export default DetectionLive;
