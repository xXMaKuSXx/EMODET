import React, { useState } from "react";
import styles from "@/styles/Card.module.css";
import Checkbox from "../Checkbox";
import Button from "../Button";
import { useRouter } from "next/router";

const Card = () => {
  const [isDetectionSelected, setIsDetectionSelected] = useState([true, false]);
  const [isModelSelected, setIsModelSelected] = useState([
    true,
    false,
    false,
    false,
  ]);

  const router = useRouter();

  const handleDetectionToggle = (index) => {
    const updatedCheckedState = isDetectionSelected.map(
      (value, i) => i === index
    );
    setIsDetectionSelected(updatedCheckedState);
  };
  const handleModelToggle = (index) => {
    const updatedCheckedState = isModelSelected.map((value, i) => i === index);
    setIsModelSelected(updatedCheckedState);
  };
  const handleStartDetection = () => {
    const currentRoute = router.asPath;
    const isLiveSelected = isDetectionSelected[0];

    let detectionPath = isLiveSelected ? "/live" : "/audio";
    let modelPath = "";

    if (isModelSelected[0]) {
      modelPath = "/model-1";
    } else if (isModelSelected[1]) {
      modelPath = "/model-2";
    } else if (isModelSelected[2]) {
      modelPath = "/model-3";
    } else {
      modelPath = "/model-4";
    }

    const finalPath = currentRoute + detectionPath + modelPath;

    router.push(finalPath);
  };

  return (
    <div className={`${styles.box}`}>
      <div className={`${styles.card}`}>
        <span className={`${styles.title}`}>CHOOSE THE DETECTION SETTINGS</span>
        <div className={`${styles.label}`}>
          Detection:
          <Checkbox
            label="Live"
            isChecked={isDetectionSelected[0]}
            onToggle={() => handleDetectionToggle(0)}
          />
          <Checkbox
            label="from Audio"
            isChecked={isDetectionSelected[1]}
            onToggle={() => handleDetectionToggle(1)}
          />
        </div>
        <div className={`${styles.label}`}>
          Model:
          <div className={`${styles.col}`}>
            <Checkbox
              label="Transformer"
              isChecked={isModelSelected[0]}
              onToggle={() => handleModelToggle(0)}
            />
            <Checkbox
              label="LSTM"
              isChecked={isModelSelected[2]}
              onToggle={() => handleModelToggle(2)}
            />
          </div>
          <div className={`${styles.col}`}>
            <Checkbox
              label="CNN"
              isChecked={isModelSelected[1]}
              onToggle={() => handleModelToggle(1)}
            />
            <Checkbox
              label="MFCC"
              isChecked={isModelSelected[3]}
              onToggle={() => handleModelToggle(3)}
            />
          </div>
        </div>
        <Button onClick={handleStartDetection}>Start Detection</Button>
      </div>
    </div>
  );
};

export default Card;
