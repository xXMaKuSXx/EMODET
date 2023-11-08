import React from "react";
import styles from "@/styles/Home.module.css";
import Card from "@/components/Card";

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
          <Card />
        </div>
      </main>
    </>
  );
};

export default DetectionPage;
