import React from "react";
import styles from "@/styles/Home.module.css";
import LiveDetectionCard from "@/components/LiveDetectionCard";

const DetectionPage = () => {
  return (
    <>
      <main className={`${styles.main}`}>
        <div className={`${styles.container}`}>
          <img
            src="/microphone.png"
            alt="Microphone"
            className={`${styles.microphone}`}
          />
          <LiveDetectionCard model="model-1" />
        </div>
      </main>
    </>
  );
};

export default DetectionPage;
