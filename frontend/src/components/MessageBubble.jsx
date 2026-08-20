import React from "react";

export default function MessageBubble({ role, message }) {
  const isUser = role === "user";
  return (
    <div className={`message-row ${isUser ? "message-row-user" : "message-row-assistant"}`}>
      <div className={`message-bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
        {message}
      </div>
    </div>
  );
}
