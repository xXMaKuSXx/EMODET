import React from "react";
import styles from "@/styles/Navbar.module.css";
import Link from "next/link";

function Navbar() {
  return (
    <nav className={`${styles.nav}`}>
      <div className={`${styles.container}`}>
        <div className="logo">
          <Link href="/">
            <img src="/Logo.png" className={`${styles.img}`}></img>
          </Link>
        </div>
        <ul className={`${styles.ul}`}>
          <li>
            <Link href="/authors">Authors</Link>
          </li>
          <li>
            <Link href="/models">Models</Link>
          </li>
          <li>
            <Link href="/detection">Detection</Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
