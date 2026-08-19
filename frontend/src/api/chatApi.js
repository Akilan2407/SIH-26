import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});
const mock = import.meta.env.VITE_USE_MOCK_API === "true";

const mockSessions = {};
const mockReply = (sessionId, message) => {
  const session = mockSessions[sessionId] || { structured_data: {}, conversation_history: [], completed: false };
  const lower = message.toLowerCase();
  if (lower.includes("year") || /\b\d{2}\b/.test(message)) session.structured_data.age = session.structured_data.age || "56 years";
  if (lower.includes("diabet")) session.structured_data.diabetes_duration = session.structured_data.diabetes_duration || "12 years";
  if (lower.includes("hba") || lower.includes("a1c")) session.structured_data.hba1c = "7.2%";
  if (lower.includes("pressure") || /\d{2,3}\/\d{2,3}/.test(message)) session.structured_data.blood_pressure = "128/82";
  const missing = ["age", "diabetes_duration", "hba1c", "blood_pressure"].filter((field) => !session.structured_data[field]);
  session.completed = missing.length === 0;
  const reply = session.completed ? "Thank you. The required information has been collected successfully." : `Thanks. Could you share your ${missing[0].replace("_", " ")} next?`;
  session.conversation_history.push({ role: "user", message }, { role: "assistant", message: reply }); mockSessions[sessionId] = session;
  return { reply, patient_state: session, completed: session.completed };
};

/**
 * Send a chat message to the backend.
 * @param {string} sessionId
 * @param {string} message
 * @returns {Promise<{reply: string, patient_state: object, completed: boolean}>}
 */
export async function sendMessage(sessionId, message) {
  if (mock) return Promise.resolve(mockReply(sessionId, message));
  try {
    const response = await client.post("/chat", {
      session_id: sessionId,
      message,
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      const detail = error.response.data?.detail || "The server returned an error.";
      throw new Error(detail);
    }
    if (error.request) {
      throw new Error("Could not reach the server. Please check your connection.");
    }
    throw new Error(error.message || "Something went wrong.");
  }
}

/**
 * Fetch the stored session state (used to restore a persisted session).
 * @param {string} sessionId
 */
export async function fetchSession(sessionId) {
  if (mock) return Promise.resolve(mockSessions[sessionId] || { conversation_history: [], structured_data: {}, completed: false });
  try {
    const response = await client.get(`/session/${sessionId}`);
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data?.detail || "Could not load session.");
    }
    throw new Error("Could not reach the server. Please check your connection.");
  }
}

export const getSession = fetchSession;
