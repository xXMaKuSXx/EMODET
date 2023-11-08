import styles from "@/styles/Home.module.css";
import Button from "@/components/Button";
import Link from "next/link";

const Home = () => {
  return (
    <>
      <main className={`${styles.main}`}>
        <div className={`${styles.container}`}>
          <img
            src="/microphone.png"
            alt="Microphone"
            className={`${styles.microphone}`}
          />
          <div className={`${styles.detectcontainer}`}>
            <h1>
              <span>Detect Emotions</span>
              <span>From Your Voice</span>
            </h1>
            <Link href="/detection">
              <Button className={`${styles.button}`}>Start Now</Button>
            </Link>
          </div>
          <img
            src="/happiness.png"
            alt="Smiling face"
            className={`${styles.smile}`}
          />
          <img
            src="/angry.png"
            alt="Angry face"
            className={`${styles.angry}`}
          />
          <img src="/sad.png" alt="Sad face" className={`${styles.sad}`} />
          <img src="/fear.png" alt="Fear face" className={`${styles.fear}`} />
        </div>
      </main>
    </>
  );
};

export default Home;
