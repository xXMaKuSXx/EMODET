import React from "react";
import styles from "@/styles/Authors.module.css";
const AuthorsPage = () => {
  return (
    <>
      <main className={`${styles.main}`}>
        <div className={`${styles.card} ${styles.left}`}>
          <span>Jakub Bieńkowski</span>
          <span>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean
            sagittis semper tellus, ac iaculis magna. Morbi laoreet libero ac
            dui egestas, sed maximus mi congue. Nullam ac felis fermentum,
            rhoncus sem quis, malesuada libero. Integer at augue in velit
            placerat condimentum. Donec ante lorem, cursus eleifend metus eu,
            fringilla mattis felis. Nunc varius dui sapien, vitae mattis nisi
            luctus sed.
          </span>
        </div>
        <div className={`${styles.card} ${styles.center}`}>
          <span>About the project</span>
          <span>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean
            sagittis semper tellus, ac iaculis magna. Morbi laoreet libero ac
            dui egestas, sed maximus mi congue. Nullam ac felis fermentum,
            rhoncus sem quis, malesuada libero. Integer at augue in velit
            placerat condimentum. Donec ante lorem, cursus eleifend metus eu,
            fringilla mattis felis. Nunc varius dui sapien, vitae mattis nisi
            luctus sed. Aenean augue sapien, ultricies et tellus gravida,
            suscipit faucibus metus. Mauris lacinia, nisi sagittis dictum
            facilisis, est metus commodo urna, quis pellentesque risus ligula
            nec metus. Pellentesque vulputate eleifend justo, vel pulvinar ipsum
            ultricies eget.
          </span>
        </div>
        <div className={`${styles.card} ${styles.right}`}>
          <span>Jakub Makowski</span>
          <span>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean
            sagittis semper tellus, ac iaculis magna. Morbi laoreet libero ac
            dui egestas, sed maximus mi congue. Nullam ac felis fermentum,
            rhoncus sem quis, malesuada libero. Integer at augue in velit
            placerat condimentum. Donec ante lorem, cursus eleifend metus eu,
            fringilla mattis felis. Nunc varius dui sapien, vitae mattis nisi
            luctus sed.
          </span>
        </div>
      </main>
    </>
  );
};

export default AuthorsPage;
