import { useState, useEffect } from "react";
import axios from "axios";
import styles from "@/styles/Card.module.css";

const EmotionDetector = (model) => {
  const [detectedEmotion, setDetectedEmotion] = useState("Unknown");
  const [emotionImage, setEmotionImage] = useState({
    src: "/detected_neutral.png",
    alt: "Neutral Emotion Image",
  });

  const detectEmotionLive = async () => {
    try {
      const response = await axios(
        `http://localhost:8080/detect-emotion-live/${model.model}`
      );
      setDetectedEmotion(response.data.detected_emotion);
      setEmotionImage(response.data.emotionImage);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  useEffect(() => {
    const interval = setInterval(() => detectEmotionLive(), 1000);
    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <div className={`${styles.result}`}>
      {detectedEmotion}
      <img
        src={emotionImage.src}
        alt={emotionImage.alt}
        className={`${styles.emotionimage}`}
      ></img>
    </div>
  );
};

export default EmotionDetector;
