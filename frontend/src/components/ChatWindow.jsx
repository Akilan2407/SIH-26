import React, { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble.jsx";
import TypingIndicator from "./TypingIndicator.jsx";

export default function ChatWindow({ messages, isTyping, error }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  return (
    <div className="chat-window">
      {messages.map((m, idx) => (
        <MessageBubble key={idx} role={m.role} message={m.message} />
      ))}
      {isTyping && <TypingIndicator />}
      {error && <div className="error-banner">{error}</div>}
      <div ref={bottomRef} />
    </div>
  );
}
