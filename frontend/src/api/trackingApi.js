import { mockTracking } from "../mock/trackingMock.js";
const mock = import.meta.env.VITE_USE_MOCK_API !== "false";
export async function getTracking() { if (mock) return mockTracking(); const response = await fetch("/tracking"); if (!response.ok) throw new Error("Unable to load model tracking data."); return response.json(); }