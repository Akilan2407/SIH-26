import { useEffect, useState } from "react";
import { ArrowRight, Bot, Check, Plus, Send, Sparkles } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { getSession, sendMessage } from "../api/chatApi.js";

const SESSION_KEY = "dr_chatbot_session_id";
const FIELDS = [
  ["age", "Age"],
  ["diabetes_duration", "Diabetes duration"],
  ["hba1c", "HbA1c"],
  ["blood_pressure", "Blood pressure"],
];

function createSessionId() {
  return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export default function Assistant() {
  const navigate = useNavigate();
  const [sessionId, setSessionId] = useState(() => localStorage.getItem(SESSION_KEY) || createSessionId());
  const [messages, setMessages] = useState([]);
  const [state, setState] = useState({ structured_data: {}, completed: false });
  const [text, setText] = useState("");
  const [typing, setTyping] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    localStorage.setItem(SESSION_KEY, sessionId);
    getSession(sessionId)
      .then((data) => {
        setMessages(data.conversation_history || []);
        setState(data);
      })
      .catch(() => setMessages([
        { role: "assistant", message: "Hi, I'm here to gather a few details before your screening. Could you start by telling me your age and how long you've had diabetes?" },
      ]));
  }, [sessionId]);

  async function submit(event) {
    event.preventDefault();
    if (!text.trim() || typing) return;
    const message = text.trim();
    setText("");
    setError("");
    setTyping(true);
    setMessages((items) => [...items, { role: "user", message }]);
    try {
      const data = await sendMessage(sessionId, message);
      setMessages(data.patient_state?.conversation_history || []);
      setState(data.patient_state || data);
    } catch (err) {
      setError(err.message || "Unable to connect to the AI assistant.");
    } finally {
      setTyping(false);
    }
  }

  function resetSession() {
    const nextId = createSessionId();
    localStorage.setItem(SESSION_KEY, nextId);
    setSessionId(nextId);
    setMessages([]);
    setState({ structured_data: {}, completed: false });
    setError("");
  }

  const collected = FIELDS.filter(([field]) => state.structured_data?.[field]).length;

  return <>
    <div className="page-heading compact">
      <div>
        <p className="eyebrow">AI ASSISTANT / INTAKE</p>
        <h1>A conversation before care</h1>
        <p className="lede">A focused intake assistant. It does not diagnose or perform screening.</p>
      </div>
      <button className="button secondary" onClick={resetSession} type="button"><Plus size={16} /> New session</button>
    </div>
    <div className="assistant-layout">
      <section className="chat-card">
        <div className="chat-head">
          <div className="assistant-avatar"><Bot size={20} /></div>
          <div><strong>RetinaCare assistant</strong><span>Secure intake · Session {sessionId.slice(-6)}</span></div>
          <span className="online">Online</span>
        </div>
        <div className="messages">
          {messages.map((item, index) => <div className={`message ${item.role === "user" ? "user" : "assistant"}`} key={`${item.message}-${index}`}>
            {item.role !== "user" && <Sparkles size={14} />}{item.message}
          </div>)}
          {typing && <div className="typing">● ● ●</div>}
          {error && <div className="inline-error">Unable to connect to the AI assistant. {error}</div>}
        </div>
        <form className="chat-form" onSubmit={submit}>
          <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Share a detail with the assistant..." disabled={typing} />
          <button aria-label="Send message" disabled={typing || !text.trim()}><Send size={17} /></button>
        </form>
      </section>
      <aside className="slots-card">
        <p className="eyebrow">INFORMATION COLLECTED</p>
        <div className="slot-count"><strong>{collected}</strong><span>/ 4 required fields</span></div>
        <div className="progress-line"><span style={{ width: `${collected * 25}%` }} /></div>
        {FIELDS.map(([field, label]) => <div className="slot" key={field}>
          <span className={state.structured_data?.[field] ? "done" : ""}>{state.structured_data?.[field] ? <Check size={13} /> : ""}</span>
          <div><strong>{label}</strong><small>{state.structured_data?.[field] || "Not collected yet"}</small></div>
        </div>)}
        {state.completed && <div className="handoff">
          <strong>Information collection complete</strong>
          <p>Your intake is ready for the screening workflow.</p>
          <button className="button primary" onClick={() => navigate("/screening")} type="button">Continue to screening <ArrowRight size={15} /></button>
        </div>}
      </aside>
    </div>
  </>;
}
