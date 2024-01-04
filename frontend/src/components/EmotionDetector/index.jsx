import { useState, useEffect } from "react";
import axios from "axios";
import styles from "@/styles/Card.module.css";

const EmotionDetector = (model) => {
  const [detectedEmotion, setDetectedEmotion] = useState("Loading...");

  const detectEmotionLive = async () => {
    try {
      const response = await axios(
        `http://localhost:8080/detect-emotion-live/${model.model}`
      );
      setDetectedEmotion(response.data.detected_emotion);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  useEffect(() => {
    const interval = setInterval(() => detectEmotionLive(), 1650);
    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <div className={`${styles.result}`}>
      {detectedEmotion}
    </div>
  );
};

export default EmotionDetector;
