import { useState } from "react";

const Checkbox = ({ label, isChecked, onToggle }) => {
  return (
    <div>
      <label>
        <input
          style={{ display: "none" }}
          type="checkbox"
          checked={isChecked}
          onChange={onToggle}
        />
        <span style={{ color: isChecked ? "#F00" : "#fff" }}>{label}</span>
      </label>
    </div>
  );
};
export default Checkbox;
