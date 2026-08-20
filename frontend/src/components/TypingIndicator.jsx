import React from "react";

export default function TypingIndicator() {
  return (
    <div className="message-row message-row-assistant">
      <div className="message-bubble bubble-assistant typing-indicator">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    </div>
  );
}
