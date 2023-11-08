import React from "react";
import styles from "@/styles/Navbar.module.css";

function Navbar() {
  return (
    <nav className={`${styles.nav}`}>
      <div className={`${styles.container}`}>
        <div className="logo">
          <a href="/">
            <img src="/Logo.png" className={`${styles.img}`}></img>
          </a>
        </div>
        <ul className={`${styles.ul}`}>
          <li>
            <a href="/authors">Authors</a>
          </li>
          <li>
            <a href="/models">Models</a>
          </li>
          <li>
            <a href="/detection">Detection</a>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
