import React, { useState } from "react";
import styles from "@/styles/Card.module.css";
import DetectionLive from "../DetectionLive";

const LiveDetectionCard = (model) => {
  return (
    <div className={`${styles.box}`}>
      <div className={`${styles.card}`}>
        <span className={`${styles.title}`}>DETECTED EMOTION</span>
        <DetectionLive model={model.model}/>
      </div>
    </div>
  );
};

export default LiveDetectionCard;
